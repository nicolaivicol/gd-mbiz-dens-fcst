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
from tsfcst.find_best_params import load_best_params
from tsfcst.forecasters.forecaster import ForecasterConfig
from tsfcst.predict_weights_rates import load_predicted
from tsfcst.time_series import TsData
from utils import get_submit_file_name
from tsfcst.find_best_weights import load_best_weights

log = logging.getLogger(os.path.basename(__file__))

# constant weights to apply to all cfips
map_best_weights = {
    'naive': {'naive': 1},
    'ma': {'ma': 1},
    'ema': {'ema': 1},
    'theta': {'theta': 1},
    'hw': {'hw': 1},
    'drift': {'drift': 1},
    'driftr': {'driftr': 1},
}


def gen_submission():
    df_submission = df_fcsts \
        .with_columns(pl.concat_str([pl.col('cfips'), pl.col('date')], sep='_').alias('row_id')) \
        .rename({'ensemble': 'microbusiness_density'}) \
        .select(['row_id', 'microbusiness_density'])

    # append revealed test to have the correct number of rows:
    df_revealed_test = pl.read_csv(f'{config.DIR_DATA}/revealed_test.csv') \
        .select(['row_id', 'microbusiness_density'])

    df_submission = pl.concat([df_submission, df_revealed_test]).sort('row_id')

    # check validity
    assert list(df_submission.columns) == ['row_id', 'microbusiness_density']
    assert len(df_submission) == 25080
    assert all(df_submission['microbusiness_density'].is_not_nan())
    assert all(df_submission['microbusiness_density'] >= 0)

    file_submission = f"{dir_out}/{get_submit_file_name(f'sub-ens-{id_run}', tag=config.VERSION)}.csv"
    df_submission.write_csv(file_submission, float_precision=4)
    log.info(f'submission saved to {file_submission}')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--target_name', default='active', help='microbusiness_density, active')
    parser.add_argument('-g', '--tag', default='ens', help='options: test, full')
    parser.add_argument('-a', '--asofdate', default='2022-07-01')
    parser.add_argument('-n', '--periodsahead', default=6, type=int, help='6 for submission')
    parser.add_argument('-s', '--selected_trials', default='best', help='options: best, top10')
    # ensembling method and best weights:
    parser.add_argument('-x', '--ensmethod', default='wavg')
    parser.add_argument('-v', '--weights', default='naive')
    # best params:
    parser.add_argument('--naive')    # 'active-20220701-naive-test-level-1-0_0'
    parser.add_argument('--ma')       # 'active-20220701-ema-test-trend_level_damp-25-0_0'
    parser.add_argument('--ema')      # 'active-20220701-ema-test-trend_level_damp-25-0_0'
    parser.add_argument('--drift')    # 'active-cfips-20221201-driftr-full-trend_level_damp-1-0_0'
    parser.add_argument('--driftr')   # 'active-cfips-20221201-driftr-full-trend_level_damp-1-0_0'
    parser.add_argument('--theta')    # 'active-cfips-20220701-theta-test-trend_level_damp-50-0_02'
    parser.add_argument('--hw')       # 'active-20220701-hw-test-trend_level_damp-100-2_0'

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    # example for single model
    # args.tag = 'model-driftr-wc1_bestpublic_locf'
    # args.asofdate = '2022-12-01'
    # args.ensmethod = 'single'
    # args.weights = 'driftr'
    # args.driftr = 'active-cfips-20221201-driftr-full-tld-1-0_0-wc1_bestpublic'

    # for submission:
    args.target = 'active'
    args.tag = 'ens'
    args.asofdate = '2022-12-01'
    args.naive = 'active-cfips-20221201-naive-full-tld-1-0_0'
    # args.ema = 'active-cfips-20221201-ema-full-tld-20-0_0'
    args.theta = 'active-cfips-20221201-theta-full-tld-50-0_02'
    args.ensmethod = 'max'
    args.weights = 'no' # 'full-weight-folds_1-active-20220701-active-target-naive_ema_theta-corner-20221201-20220801-manual_fix'

    log.info(f'Running {os.path.basename(__file__)} with parameters: \n'
             + json.dumps(vars(args), indent=2))
    log.info('This forecasts using the ensemble of several models (naive, ma, theta, hw) '
             'combined with the best weights found. Each of the model is initialized with '
             'the best params found based on cv.')

    allowed_model_names = ['naive', 'ma', 'ema', 'drift', 'driftr', 'theta', 'hw']
    dict_best_params_run_id = {}
    for m in allowed_model_names:
        if vars(args).get(m, None) is not None:
            dict_best_params_run_id[m] = vars(args)[m]
    model_names = list(dict_best_params_run_id.keys())
    assert len(model_names) > 0
    log.debug(f"model_names={', '.join(model_names)}")

    target_name = args.target_name
    tag = args.tag
    asofdate = args.asofdate
    periodsahead = args.periodsahead
    ensmethod = args.ensmethod
    weights_name = args.weights
    selected_trials = args.selected_trials

    id_run = f"{tag}-{'_'.join(model_names)}-{asofdate.replace('-', '')}-{target_name}-{ensmethod}-{weights_name}"
    dir_out = f'{config.DIR_ARTIFACTS}/{Path(__file__).stem}/{id_run}'
    os.makedirs(dir_out, exist_ok=True)

    log.debug('load time series data')
    df_train, df_test, df_census, df_pop = load_data()
    if asofdate is not None:
        df_train = df_train.filter(pl.col('first_day_of_month') <= pl.lit(asofdate).str.strptime(pl.Date, fmt='%Y-%m-%d'))
    list_cfips = sorted(np.unique(df_train['cfips']))
    log.debug(f'{len(list_cfips)} cfips loaded')

    log.debug('loading best params')
    dict_df_best_params = {}
    for model_name_, id_best_params in dict_best_params_run_id.items():
        df_best_params = load_best_params(id_best_params, selected_trials)
        dict_df_best_params[model_name_] = df_best_params

    log.debug('try loading data frame with weights per cfips')
    try:
        df_best_weights = load_best_weights(weights_name)
    except ValueError as e:
        try:
            df_best_weights = load_predicted(weights_name)
        except Exception as e:
            df_best_weights = None

    best_weights = map_best_weights.get(weights_name, None)

    if ensmethod in ['weighted_average', 'wavg']:
        assert df_best_weights is not None or best_weights is not None

    if df_best_weights is not None:
        assert all([m in df_best_weights.columns for m in model_names])
    elif best_weights is not None:
        assert all([m in best_weights.keys() for m in model_names])

    log.debug('generate forecast for each cfips')
    list_fcsts = []
    for cfips in tqdm(list_cfips, unit='cfips', leave=False, mininterval=3):
        # load time series for cfips
        df_ts = get_df_ts_by_cfips(cfips, target_name, df_train)
        ts = TsData(df_ts['first_day_of_month'], df_ts[target_name])

        # load best weights
        if df_best_weights is not None:
            best_weights = df_best_weights.filter(pl.col('cfips') == cfips).select(model_names).to_dicts()[0]

        # load configs with best params of several forecasters
        best_cfgs = {}
        for name in model_names:
            df_best_params = dict_df_best_params[name]
            cfg = ForecasterConfig.from_df_with_best_params(cfips, df_best_params)
            best_cfgs[name] = cfg

        ens = Ensemble(data=ts, fcster_configs=best_cfgs, method=ensmethod, weights=best_weights)
        fcst = ens.forecast(periodsahead)
        fcst = fcst.with_columns(pl.lit(cfips).cast(pl.Int64).alias('cfips'))
        list_fcsts.append(fcst)

    df_fcsts = pl.concat(list_fcsts).with_columns(pl.col('cfips'))

    if target_name == 'active':
        df_fcsts = df_fcsts.join(df_pop.rename({'first_day_of_month': 'date'}), on=['cfips', 'date'], how='left')
        for name in (model_names + ['ensemble']):
            df_fcsts = df_fcsts.with_columns((pl.col(name) / pl.col('population') * 100).alias(name))

    file_fcsts = f'{dir_out}/fcsts_all_models.csv'
    df_fcsts.write_csv(file_fcsts)
    log.info(f'all forecasts saved to {file_fcsts}')

    if asofdate == '2022-12-01':
        log.debug(f'as of date = {asofdate}, generate submission...')
        gen_submission()

    log.info(f'Run: "{id_run}" by {os.path.basename(__file__)} finished successfully.')
