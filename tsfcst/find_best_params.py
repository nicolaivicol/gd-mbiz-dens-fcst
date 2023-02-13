import os
from pathlib import Path
import logging
import argparse
from tqdm import tqdm
import json
import numpy as np
import pandas as pd
import polars as pl

import config
from etl import load_data, get_ts_by_cfips
from tsfcst.params_finder import ParamsFinder
from tsfcst.models.inventory import MODELS
from tsfcst.time_series import TsData


log = logging.getLogger(os.path.basename(__file__))


def get_id_run(target_name, asofdate, model, cv_args, search_args, n_trials, reg_coef, **kwargs):
    asofdate = pd.to_datetime(asofdate).strftime('%Y%m%d')
    reg_coef = str(reg_coef).replace('.', '_')
    return f"{target_name}-{asofdate}-{model}-{cv_args}-{search_args}-{n_trials}-{reg_coef}"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_name', default='microbusiness_density', help='options: microbusiness_density, active')
    parser.add_argument('--model', default='theta', help='options: ma, theta, hw, prophet')
    parser.add_argument('--cv_args', default='test', help='options: test, full')
    parser.add_argument('--search_args', default='trend_level_damp')
    parser.add_argument('--n_trials', default=100, type=int)
    parser.add_argument('--reg_coef', default=0.0, type=float)
    parser.add_argument('--asofdate', default='2022-07-01')
    args = parser.parse_args()
    args_as_dict = vars(args)

    # python -m tsfcst.find_best_params --target_name microbusiness_density --model theta --cv_args test --n_trials 100 --reg_coef 0 --asofdate 2022-07-01

    id_run = get_id_run(**vars(args))
    log.info(f'Running {os.path.basename(__file__)}, id_run={id_run}, with parameters: \n'
             + json.dumps(args_as_dict, indent=2))
    log.info('This finds best hyper-parameters of the model based on CV via optuna, ETA ~???min.')

    target_name = args.target_name
    model = args.model
    n_trials = args.n_trials
    cv_args = config.CV_ARGS_DICT[args.cv_args]
    search_args = config.SEARCH_ARGS_DICT[args.search_args]
    reg_coef = args.reg_coef
    asofdate = args.asofdate

    dir_out = f'{config.DIR_ARTIFACTS}/{Path(__file__).stem}/{id_run}'
    os.makedirs(dir_out, exist_ok=True)

    # load data
    df_train, df_test, df_census = load_data()
    df_train = df_train.filter(pl.col('first_day_of_month') <= pd.to_datetime(asofdate))
    list_cfips = sorted(np.unique(df_train['cfips']))
    log.debug(f'{len(list_cfips)} cfips loaded')

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

    for cfips in tqdm(list_cfips[:10], unit='cfips'):
        df_ts = get_ts_by_cfips(cfips, target_name, df_train)
        ParamsFinder.data = TsData(df_ts['first_day_of_month'], df_ts[target_name])

        df_trials, best_result, param_importances = ParamsFinder.find_best(n_trials=n_trials, use_cache=False, param_importances=False)
        print('best_params: \n' + str(best_result))

        res_best = {
            **{'cfips': cfips},
            **args_as_dict,
            **{ParamsFinder.name_objective: best_result['best_value'], 'selected_trials': 'best'},
            **best_result['best_params']
        }

        best_metric, best_params_median = ParamsFinder.best_params_top_median(df_trials, max_combs=10)

        res_median = {
            **{'cfips': cfips},
            **args_as_dict,
            **{ParamsFinder.name_objective: best_metric, 'selected_trials': 'top10'},
            **best_params_median
        }

        pl.DataFrame([res_best, res_median]).write_csv(f'{dir_out}/{cfips}.csv', float_precision=5)

        # 132223 ms
