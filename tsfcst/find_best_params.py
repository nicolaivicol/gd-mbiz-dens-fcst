import os
from pathlib import Path
import logging
import argparse
from tqdm import tqdm
import json
import numpy as np
import pandas as pd
import polars as pl
import time

import config
from etl import load_data, get_df_ts_by_cfips
from tsfcst.params_finder import ParamsFinder
from tsfcst.models.inventory import MODELS
from tsfcst.time_series import TsData


log = logging.getLogger(os.path.basename(__file__))


def eta(n_cfips, model, n_parts, n_trials):
    n_parts = 1 if n_parts is None else n_parts
    sec_per_cfips_100_trials_map = {
        1: {'ma': 4, 'theta': 5.2, 'hw': 5.3},
        4: {'ma': 5, 'theta': 8.2, 'hw': 9.0},
        8: {'ma': 6, 'theta': 10.0, 'hw': 11.0},
        16: {'ma': 15, 'theta': 24.0, 'hw': 26.0}
    }
    sec_per_cfips_100_trials = sec_per_cfips_100_trials_map \
        .get(n_parts, sec_per_cfips_100_trials_map[1]) \
        .get(model, 5.5)
    return n_cfips * sec_per_cfips_100_trials * n_trials / 100


def add_common_args(parser):
    parser.add_argument('-t', '--target_name', default='microbusiness_density',
                        help='options: microbusiness_density, active')
    parser.add_argument('-m', '--model', default='theta',
                        help='options: ma, theta, hw, prophet')
    parser.add_argument('-cv', '--cv_args', default='test',
                        help='options: test, full')
    parser.add_argument('-s', '--search_args', default='trend_level_damp')
    parser.add_argument('-nt', '--n_trials', default=100, type=int)
    parser.add_argument('-rc', '--reg_coef', default=0.0, type=float)
    parser.add_argument('-aod', '--asofdate', default='2022-07-01')
    return parser


def parse_args():
    parser = argparse.ArgumentParser()
    parser = add_common_args(parser)
    parser.add_argument('-p', '--part', type=int)
    parser.add_argument('-np', '--n_parts', type=int)
    args = parser.parse_args()
    return args


def get_id_run(target_name, asofdate, model, cv_args, search_args, n_trials, reg_coef, **kwargs):
    asofdate = pd.to_datetime(asofdate).strftime('%Y%m%d')
    reg_coef = str(reg_coef).replace('.', '_')
    return f"{target_name}-{asofdate}-{model}-{cv_args}-{search_args}-{n_trials}-{reg_coef}"


if __name__ == '__main__':
    args = parse_args()

    # python -m tsfcst.find_best_params -t microbusiness_density -m theta -cv test -nt 100 -rc 0 -aod 2022-07-01 -p 1 -np 16

    id_run = get_id_run(**vars(args))
    log.info(f'Running {os.path.basename(__file__)}, id_run={id_run}, with parameters: \n'
             + json.dumps(vars(args), indent=2))
    log.info('This finds best hyper-parameters of the model based on CV via optuna.')

    target_name = args.target_name
    model = args.model
    n_trials = args.n_trials
    cv_args = config.CV_ARGS_DICT[args.cv_args]
    search_args = config.SEARCH_ARGS_DICT[args.search_args]
    reg_coef = args.reg_coef
    asofdate = args.asofdate
    part = args.part
    n_parts = args.n_parts

    dir_out = f'{config.DIR_ARTIFACTS}/{Path(__file__).stem}/{id_run}'
    os.makedirs(dir_out, exist_ok=True)

    # load data
    df_train, df_test, df_census = load_data()
    df_train = df_train.filter(pl.col('first_day_of_month') <= pd.to_datetime(asofdate))
    list_cfips = sorted(np.unique(df_train['cfips']))
    n_cfips = len(list_cfips)

    if part is not None and n_parts is not None:
        # take partition
        step = int(n_cfips / n_parts) + 1
        i_start = (part - 1) * step
        i_end = i_start + step
        list_cfips = list_cfips[i_start:i_end]

    log.debug(f'{len(list_cfips)} cfips loaded out of {n_cfips} available')
    eta_ = eta(len(list_cfips), model, n_parts, n_trials)
    log.info(f'ETA is {time.strftime("%Hh %Mmin %Ssec", time.gmtime(eta_))}')

    # settings of parameters finder
    ParamsFinder.model_cls = MODELS[model]
    ParamsFinder.trend = search_args['trend']
    ParamsFinder.seasonal = search_args['seasonal']
    ParamsFinder.multiplicative = search_args['multiplicative']
    ParamsFinder.level = search_args['level']
    ParamsFinder.damp = search_args['damp']
    ParamsFinder.reg_coef = reg_coef
    ParamsFinder.n_train_dates = cv_args['n_train_dates']
    ParamsFinder.step_train_dates = cv_args['step_train_dates']
    ParamsFinder.periods_val = cv_args['periods_val']
    ParamsFinder.periods_test = cv_args['periods_test']
    ParamsFinder.periods_out = cv_args['periods_out']
    ParamsFinder.periods_val_last = cv_args['periods_val_last']

    for cfips in tqdm(list_cfips, unit='cfips'):
        log.debug(f'cfips={cfips}')
        df_ts = get_df_ts_by_cfips(cfips, target_name, df_train)
        ParamsFinder.data = TsData(df_ts['first_day_of_month'], df_ts[target_name])

        try:
            df_trials, best_result, param_importances = ParamsFinder.find_best(
                n_trials=n_trials,
                use_cache=False,
                parimp=False
            )
        except Exception as e:
            log.error(f'ParamsFinder.find_best() failed for {cfips}')
            continue

        res_best = {
            **{'cfips': cfips},
            **vars(args),
            **{ParamsFinder.name_objective: best_result['best_value'], 'selected_trials': 'best'},
            **best_result['best_params']
        }

        best_metric, best_params_median = ParamsFinder.best_params_top_median(df_trials, max_combs=10)

        res_median = {
            **{'cfips': cfips},
            **vars(args),
            **{ParamsFinder.name_objective: best_metric, 'selected_trials': 'top10'},
            **best_params_median
        }

        pl.DataFrame([res_best, res_median]).write_csv(f'{dir_out}/{cfips}.csv', float_precision=5)
