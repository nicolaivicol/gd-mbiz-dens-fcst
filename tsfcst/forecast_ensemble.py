import os
import glob
from pathlib import Path
import logging
import argparse
from tqdm import tqdm
import json
import numpy as np
import polars as pl

import config
from etl import load_data, get_df_ts_by_cfips
from tsfcst.ensemble import Ensemble
from tsfcst.forecasters.forecaster import ForecasterConfig
from tsfcst.time_series import TsData
from utils import get_submit_file_name

log = logging.getLogger(os.path.basename(__file__))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--target_name', default='microbusiness_density', help='microbusiness_density, active')
    parser.add_argument('-g', '--tag', default='test_w_ideal_weights', help='options: test, full')
    parser.add_argument('-a', '--asofdate', default='2022-07-01')
    parser.add_argument('-n', '--periodsahead', default=3, type=int, help='3 for test, 8 for full')
    parser.add_argument('-s', '--selected_trials', default='best', help='options: best, top10')
    # best params:
    parser.add_argument('-i', '--naive', default='microbusiness_density-20220701-naive-test-trend_level_damp-1-0_0')
    parser.add_argument('-o', '--ma', default='microbusiness_density-20220701-ma-test-trend_level_damp-100-0_0')
    parser.add_argument('-e', '--theta', default='microbusiness_density-20220701-theta-test-trend_level_damp-100-0_0')
    parser.add_argument('-w', '--hw', default='microbusiness_density-20220701-hw-test-trend_level_damp-100-0_0')
    # best weights:
    parser.add_argument('-v', '--weights', default='microbusiness_density-test-20221001')  # microbusiness_density-cv-20220701, microbusiness_density-test-20221001

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    log.info(f'Running {os.path.basename(__file__)} with parameters: \n'
             + json.dumps(vars(args), indent=2))
    log.info('This forecasts using the ensemble of several models (naive, ma, theta, hw) '
             'combined with the best weights found. Each of the model is initialized with '
             'the best params found based on cv.')

    target_name = args.target_name
    tag = args.tag
    asofdate = args.asofdate
    periodsahead = args.periodsahead
    dict_best_params_run_id = {'naive': args.naive, 'ma': args.ma, 'theta': args.theta, 'hw': args.hw}
    model_names = list(dict_best_params_run_id.keys())
    weights_name = args.weights
    selected_trials = args.selected_trials

    id_run = f"{target_name}-{tag}-{asofdate.replace('-', '')}"
    dir_out = f'{config.DIR_ARTIFACTS}/{Path(__file__).stem}/{id_run}'
    os.makedirs(dir_out, exist_ok=True)

    log.debug('load time series data')
    df_train, _, _ = load_data()
    if asofdate is not None:
        df_train = df_train.filter(pl.col('first_day_of_month') <= pl.lit(asofdate).str.strptime(pl.Date, fmt='%Y-%m-%d'))
    list_cfips = sorted(np.unique(df_train['cfips']))
    log.debug(f'{len(list_cfips)} cfips loaded')

    log.debug('loading best params')
    dict_df_best_params = {}
    for name, id_run in dict_best_params_run_id.items():
        dir_best_params = f'{config.DIR_ARTIFACTS}/find_best_params/{id_run}'
        files_best_params = glob.glob(f'{dir_best_params}/*.csv')
        df_best_params = pl.concat([pl.read_csv(f) for f in files_best_params])
        df_best_params = df_best_params.filter(pl.col('selected_trials') == selected_trials)
        dict_df_best_params[name] = df_best_params

    log.debug('loading best weights')
    dir_best_weights = f'{config.DIR_ARTIFACTS}/find_best_weights/{weights_name}'
    files_best_weights = glob.glob(f'{dir_best_weights}/*.csv')
    df_best_weights = pl.concat([pl.read_csv(f) for f in glob.glob(f'{dir_best_weights}/*.csv')])

    log.debug('generate forecast for each cfips')
    list_fcsts = []
    for cfips in tqdm(list_cfips, unit='cfips', leave=False, mininterval=3):
        # load time series for cfips
        df_ts = get_df_ts_by_cfips(cfips, target_name, df_train)
        ts = TsData(df_ts['first_day_of_month'], df_ts[target_name])

        # load configs with best params of several forecasters
        best_cfgs = {}
        for name in model_names:
            df_best_params = dict_df_best_params[name]
            cfg = ForecasterConfig.from_df_with_best_params(cfips, df_best_params)
            best_cfgs[name] = cfg

        # load best weights
        best_weights = df_best_weights.filter(pl.col('cfips') == cfips).select(model_names).to_dicts()[0]

        ens = Ensemble(data=ts, fcster_configs=best_cfgs, weights=best_weights)
        fcst = ens.forecast(periodsahead)
        fcst = fcst.with_columns(pl.lit(cfips).alias('cfips'))
        list_fcsts.append(fcst)

    df_fcsts = pl.concat(list_fcsts)
    file_fcsts = f'{dir_out}/fcsts_all_models.csv'
    df_fcsts.write_csv(file_fcsts)
    log.info(f'all forecasts saved to {file_fcsts}')

    df_submission = df_fcsts \
        .with_columns(pl.concat_str([pl.col('cfips'), pl.col('date')], sep='_').alias('row_id')) \
        .rename({'ensemble': 'microbusiness_density'}) \
        .select(['row_id', 'microbusiness_density'])

    file_submission = f"{dir_out}/{get_submit_file_name('submission-ens', tag=config.VERSION)}.csv"
    df_submission.write_csv(file_submission, float_precision=4)
    log.info(f'submission saved to {file_submission}')
