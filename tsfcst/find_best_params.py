import os
import glob
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
from etl import load_data, get_df_ts_by_id
from tsfcst.params_finder import ParamsFinder
from tsfcst.models.inventory import MODELS
from tsfcst.time_series import TsData
from tsfcst.utils_tsfcst import get_lin_reg_summary, chowtest

log = logging.getLogger(os.path.basename(__file__))


def eta(n_ids, model, nparts, ntrials):
    nparts = 1 if nparts is None else nparts
    sec_per_id_100_trials_map = {
        1: {'ma': 4, 'theta': 5.2, 'hw': 5.3},
        4: {'ma': 5, 'theta': 8.2, 'hw': 9.0},
        8: {'ma': 6, 'theta': 10.0, 'hw': 11.0},
        16: {'ma': 15, 'theta': 24.0, 'hw': 26.0}
    }
    sec_per_id_100_trials = sec_per_id_100_trials_map \
        .get(nparts, sec_per_id_100_trials_map[1]) \
        .get(model, 5.5)
    return n_ids * sec_per_id_100_trials * ntrials / 100


def add_common_args(parser):
    parser.add_argument('-t', '--targetname', default='active', help='microbusiness_density, active')
    parser.add_argument('-i', '--idcol', default='cfips')
    parser.add_argument('-m', '--model', default='theta', help='ma, ema, theta, hw, prophet')
    parser.add_argument('-c', '--cvargs', default='full', help='options: test, full')
    parser.add_argument('-s', '--searchargs', default='tld')
    parser.add_argument('-n', '--ntrials', default=100, type=int)
    parser.add_argument('-r', '--regcoef', default=0.0, type=float)
    parser.add_argument('-a', '--asofdate', default='2022-07-01')
    parser.add_argument('-g', '--tag', default='')
    return parser


def parse_args():
    parser = argparse.ArgumentParser()
    parser = add_common_args(parser)
    parser.add_argument('-p', '--part', type=int)
    parser.add_argument('-x', '--nparts', type=int)
    args = parser.parse_args()
    return args


def get_id_run(targetname, asofdate, model, cvargs, searchargs, ntrials, regcoef, idcol, tag, **kwargs):
    asofdate = pd.to_datetime(asofdate).strftime('%Y%m%d')
    regcoef = str(regcoef).replace('.', '_')
    id = f"{targetname}-{idcol}-{asofdate}-{model}-{cvargs}-{searchargs}-{ntrials}-{regcoef}"
    if tag is not None and tag != '':
        id += f'-{tag}'
    return id


def load_best_params(id_best_params, selected_trials='best'):
    dir_best_params = f'{config.DIR_ARTIFACTS}/find_best_params/{id_best_params}'
    files_best_params = glob.glob(f'{dir_best_params}/*.csv')
    df_best_params = pl.concat([pl.read_csv(f) for f in files_best_params], how='diagonal')
    df_best_params = df_best_params.filter(pl.col('selected_trials') == selected_trials)
    return df_best_params


if __name__ == '__main__':
    args = parse_args()

    # python -m tsfcst.find_best_params -t active -m hw -c test -s level -n 100 -r 0 -a 2022-12-01 -i cfips -p 1 -x 16

    # run in parallel
    # ./tsfcst/find_best_params.sh -t active -m naive -c test -s trend_level_damp -n 20 -r 0 -a 2022-07-01 -i cfips
    # ./tsfcst/find_best_params.sh -t active -m naive -c full -s trend_level_damp -n 20 -r 0 -a 2022-12-01 -i cfips

    # ./tsfcst/find_best_params.sh -t active -m ema -c test -s trend_level_damp -n 20 -r 0 -a 2022-07-01 -i cfips
    # ./tsfcst/find_best_params.sh -t active -m ema -c full -s trend_level_damp -n 20 -r 0 -a 2022-12-01 -i cfips

    # ./tsfcst/find_best_params.sh -t active -m theta -c test -s tld -n 50 -r 0.02 -a 2022-07-01 -i cfips
    # ./tsfcst/find_best_params.sh -t active -m theta -c full -s tld -n 50 -r 0.02 -a 2022-12-01 -i cfips

    # python -m tsfcst.find_best_params -t active -m driftr -c test -s trend_level_damp -n 1 -r 0 -a 2022-07-01 -i cfips -g ws050_wct050_m150
    # python -m tsfcst.find_best_params -t active -m driftr -c full -s trend_level_damp -n 1 -r 0 -a 2022-12-01 -i cfips -g ws050_wct050_m150

    # by state, for growth rates
    # python -m tsfcst.find_best_params -t active -m theta -c test -s trend_level_damp -n 100 -r 0.02 -a 2022-07-01 -i state
    # python -m tsfcst.find_best_params -t active -m theta -c full -s trend_level_damp -n 100 -r 0.02 -a 2022-12-01 -i state

    id_run = get_id_run(**vars(args))
    log.info(f'Running {os.path.basename(__file__)}, id_run={id_run}, with parameters: \n'
             + json.dumps(vars(args), indent=2))
    log.info('This finds best hyper-parameters of the model based on CV via optuna.')

    targetname = args.targetname
    model = args.model
    ntrials = args.ntrials
    cvargs = config.CV_ARGS_DICT[args.cvargs]
    searchargs = config.SEARCH_ARGS_DICT[args.searchargs]
    regcoef = args.regcoef
    asofdate = args.asofdate
    part = args.part
    nparts = args.nparts
    idcol = args.idcol

    dir_out = f'{config.DIR_ARTIFACTS}/{Path(__file__).stem}/{id_run}'
    os.makedirs(dir_out, exist_ok=True)

    # load data
    df_train, df_test, df_census, df_pop = load_data()
    df_train = df_train.filter(pl.col('first_day_of_month') <= pd.to_datetime(asofdate))
    list_ids = sorted(np.unique(df_train[idcol]))
    if idcol == 'state':
        list_ids.insert(0, 'all')
    n_ids = len(list_ids)

    if part is not None and nparts is not None:
        # take partition
        step = int(n_ids / nparts) + 1
        i_start = (part - 1) * step
        i_end = i_start + step
        list_ids = list_ids[i_start:i_end]

    log.debug(f'{len(list_ids)} {idcol}s loaded out of {n_ids} available')
    eta_ = eta(len(list_ids), model, nparts, ntrials)
    log.info(f'ETA is {time.strftime("%Hh %Mmin %Ssec", time.gmtime(eta_))}')

    # settings of parameters finder
    ParamsFinder.model_cls = MODELS[model]
    ParamsFinder.trend = searchargs['trend']
    ParamsFinder.seasonal = searchargs['seasonal']
    ParamsFinder.multiplicative = searchargs['multiplicative']
    ParamsFinder.level = searchargs['level']
    ParamsFinder.damp = searchargs['damp']
    ParamsFinder.reg_coef = regcoef
    ParamsFinder.n_train_dates = cvargs['n_train_dates']
    ParamsFinder.step_train_dates = cvargs['step_train_dates']
    ParamsFinder.periods_val = cvargs['periods_val']
    ParamsFinder.periods_test = cvargs['periods_test']
    ParamsFinder.periods_out = cvargs['periods_out']
    ParamsFinder.periods_val_last = cvargs['periods_val_last']

    is_simple_model = model in ['naive', 'ma', 'ema', 'sma', 'drift']
    if is_simple_model:
        ParamsFinder.trend = False
        ParamsFinder.seasonal = False
        ParamsFinder.multiplicative = False
        ParamsFinder.level = False
        ParamsFinder.damp = False
        ParamsFinder.choices_use_data_since = ['all']

        if model == 'drift':
            ParamsFinder.trend = True

    for id_ts in tqdm(list_ids, unit=idcol):
        log.debug(f'{idcol}={id_ts}')
        df_ts = get_df_ts_by_id(idcol, id_ts, targetname, df_train)
        ts = TsData(df_ts['first_day_of_month'], df_ts[targetname])
        ParamsFinder.data = ts

        if not is_simple_model:
            # reset defaults first...
            ParamsFinder.trend = searchargs['trend']
            ParamsFinder.choices_use_data_since = None

            # then override settings based on data:
            # - use data after structural break
            lastidx_20210101 = np.where(ts.time == pd.to_datetime('2021-01-01'))[0][0]
            chow_value, p_value = chowtest(ts.target, lastidx_20210101)
            if chow_value > 100:
                ParamsFinder.choices_use_data_since = ['2021-02-01']

            # - do not look for trends if last 18 obs do not exhibit a trend
            if ParamsFinder.trend:
                lin_reg_summary_last_18 = get_lin_reg_summary(ts.target, 18)
                slope_time = lin_reg_summary_last_18['slope']
                pval_time = lin_reg_summary_last_18['p_value']
                if abs(slope_time) < 0.001 or pval_time > 0.10 or (abs(slope_time) < 0.002 and pval_time > 0.05):
                    ParamsFinder.trend = False

        try:
            df_trials, best_result, param_importances = ParamsFinder.find_best(
                n_trials=ntrials,
                use_cache=False,
                parimp=False
            )
        except Exception as e:
            log.error(f'ParamsFinder.find_best() failed for {id_ts}')
            continue

        res_best = {
            **{idcol: id_ts},
            **vars(args),
            **{ParamsFinder.name_objective: best_result['best_value'], 'selected_trials': 'best'},
            **best_result['best_params']
        }

        # best_metric, best_params_median = ParamsFinder.best_params_top_median(df_trials, max_combs=10)
        #
        # res_median = {
        #     **{'cfips': cfips},
        #     **vars(args),
        #     **{ParamsFinder.name_objective: best_metric, 'selected_trials': 'top10'},
        #     **best_params_median
        # }

        pl.DataFrame([res_best]).write_csv(f'{dir_out}/{id_ts}.csv', float_precision=5)
