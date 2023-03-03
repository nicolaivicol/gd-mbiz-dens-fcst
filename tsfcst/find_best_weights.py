import os
from pathlib import Path
import logging
import argparse
from tqdm import tqdm
import json
import numpy as np
import polars as pl
import glob

import config
from etl import load_data
from tsfcst.weights_finder import WeightsFinder
from tsfcst.models.inventory import MODELS


log = logging.getLogger(os.path.basename(__file__))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--target_name', default='active', help='active, microbusiness_density')
    parser.add_argument('-g', '--tag', default='cv', help='options: cv, test, full')
    parser.add_argument('-a', '--asofdate', default='2022-07-01')
    parser.add_argument('-f', '--fromdate', default='no')
    parser.add_argument('-m', '--method', default='best', help='best, errors, corner')
    parser.add_argument('-n', '--ntrials', default=100, type=int)
    # sources of forecasts on validation / test folds
    parser.add_argument('-i', '--naive')   #, default='active-20221201-naive-full-trend_level_damp-1-0_0')  # active-20220701-naive-test-level-1-0_0
    parser.add_argument('-o', '--ma')      #, default='active-20221201-ema-full-trend_level_damp-20-0_0')  # active-20220701-ema-test-trend_level_damp-25-0_0
    parser.add_argument('-k', '--ema')
    parser.add_argument('-d', '--driftr')
    parser.add_argument('-e', '--theta')   #, default='active-20221201-theta-full-trend_level_damp-50-0_0')  # active-20220701-theta-test-trend_level_damp-50-0_0
    parser.add_argument('-w', '--hw')      #, default='')
    # partition
    parser.add_argument('-p', '--part', type=int)
    parser.add_argument('-x', '--nparts', type=int)
    args = parser.parse_args()
    return args


def load_best_weights(weights_id, model_names = None, normalize=True):
    dir_best_weights = f'{config.DIR_ARTIFACTS}/find_best_weights/{weights_id}'
    files_best_weights = glob.glob(f'{dir_best_weights}/*.csv')
    if len(files_best_weights) == 0:
        raise ValueError(f'files not found in {dir_best_weights}')
    df_best_weights = pl.concat([pl.read_csv(f) for f in files_best_weights])

    # normalize to have sum of weights = 1:
    if normalize:
        if model_names is None:
            model_names = [m for m in df_best_weights.columns if m in MODELS.keys()]
        df_best_weights = df_best_weights.with_columns(pl.sum(model_names).alias('sum_weights'))
        for model_name in model_names:
            df_best_weights = df_best_weights.with_columns(pl.col(model_name) / pl.col('sum_weights'))
        df_best_weights = df_best_weights.drop('sum_weights')

    return df_best_weights


if __name__ == '__main__':
    args = parse_args()

    log.info(f'Running {os.path.basename(__file__)} with parameters: \n'
             + json.dumps(vars(args), indent=2))
    log.info('This finds best weights to apply when ensembling the models: naive, ma, theta, hw.')

    allowed_model_names = ['naive', 'ma', 'ema', 'drift', 'driftr', 'theta', 'hw']
    dict_models_id_run = {}
    for m in allowed_model_names:
        id_m = vars(args).get(m, None)
        if id_m is not None and id_m not in ['', ' ', 'no', 'none', 'x']:
            dict_models_id_run[m] = id_m
    model_names = list(dict_models_id_run.keys())
    assert len(model_names) > 0
    log.debug(f"model_names={', '.join(model_names)}")

    target_name = args.target_name
    tag = args.tag
    asofdate = args.asofdate
    fromdate = args.fromdate
    ntrials = args.ntrials
    method = args.method
    part = args.part
    nparts = args.nparts

    id_run = f"{target_name}-{tag}-{'_'.join(model_names)}-{method}-{asofdate.replace('-', '')}-{fromdate.replace('-', '')}"
    dir_out = f'{config.DIR_ARTIFACTS}/{Path(__file__).stem}/{id_run}'
    os.makedirs(dir_out, exist_ok=True)

    # load forecasts on validation
    df_fcsts: pl.DataFrame = None

    for model_name, id_run_model in dict_models_id_run.items():
        file_df_fcsts_cv = f'{config.DIR_ARTIFACTS}/test_best_params/{id_run_model}/df_fcsts_cv.csv'
        df_model = pl.read_csv(file_df_fcsts_cv, parse_dates=True)

        # stack all folds into one column
        cols_fcsts = [col for col in df_model.columns if col.startswith('fcst_')]
        dfs_folds = []
        for col_fcst in cols_fcsts:
            df_fold = df_model \
                .select(['cfips', 'date', col_fcst]) \
                .rename({col_fcst: model_name}) \
                .with_columns(pl.lit(col_fcst).alias('fold'))
            dfs_folds.append(df_fold)

        df_stacked = pl.concat(dfs_folds)
        df_stacked = df_stacked.filter(pl.col(model_name).is_not_null())

        if df_fcsts is None:
            df_fcsts = df_stacked
        else:
            df_fcsts = df_fcsts.join(df_stacked, on=['cfips', 'fold', 'date'])

    df_fcsts = df_fcsts.sort(['cfips', 'date', 'fold'])

    if asofdate is not None:
        df_fcsts = df_fcsts.filter(pl.col('date') <= pl.lit(asofdate).str.strptime(pl.Date, fmt='%Y-%m-%d'))

    if fromdate is not None and fromdate.lower() not in ['none', 'null', 'na', '', 'nan', 'no', 'x']:
        df_fcsts = df_fcsts.filter(pl.col('date') >= pl.lit(fromdate).str.strptime(pl.Date, fmt='%Y-%m-%d'))

    log.debug('count of dates: \n' + str(df_fcsts.select(['cfips', 'date']).groupby(['cfips']).count().describe()))

    # load actual data
    df_actual, _, _, _ = load_data()
    df_actual = df_actual.rename({'first_day_of_month': 'date'}).select(['cfips', 'date', target_name])

    # merge forecasts and actuals
    df = df_fcsts.join(df_actual, on=['cfips', 'date'], how='left')

    list_cfips = sorted(np.unique(df['cfips']))
    log.debug(f'{len(list_cfips)} cfips loaded')

    if part is not None and nparts is not None:
        # take partition
        step = int(len(list_cfips) / nparts) + 1
        i_start = (part - 1) * step
        i_end = i_start + step
        list_cfips = list_cfips[i_start:i_end]
        log.debug(f'partition {part} out of {nparts}: {len(list_cfips)} cfips to run')

    list_res = []

    for cfips in tqdm(list_cfips, unit='cfips', mininterval=30, miniters=10, leave=False):
        log.debug(f'cfips={cfips}')

        df_cfips = df.filter(pl.col('cfips') == cfips)
        WeightsFinder.y_true = df_cfips[target_name].to_numpy()
        WeightsFinder.y_preds = df_cfips[model_names].to_numpy()
        WeightsFinder.model_names = model_names

        if method in ['find_best', 'best']:
            res = WeightsFinder.find_best(n_trials=ntrials)
        elif method in ['find_from_errors', 'errors']:
            res = WeightsFinder.find_from_errors()
        elif method in ['find_best_corner', 'corner']:
            res = WeightsFinder.find_best_corner()
        else:
            raise ValueError(f'method={method} is not supported')

        msg_str = ' | '.join([f"{name_}={w:.2f}" for name_, w in res['best_params'].items()])
        log.debug(f'best weights: ' + msg_str)

        smape_models = {f'smape_{m}': WeightsFinder.smape(w)
                        for m, w in zip(model_names, WeightsFinder.params_corners())}
        msg_str = ' | '.join([f"best={res['smape']:.2f}"]
                             + [f'{name_}={smape_:.2f}' for name_, smape_ in zip(model_names, smape_models.values())])
        log.debug(f'compare SMAPEs:  ' + msg_str)

        res_dict = {
            **{'cfips': cfips, 'smape': np.round(res['smape'], 4)},
            **smape_models,
            **res['best_params'],
        }
        list_res.append(res_dict)

    df_res = pl.DataFrame(list_res)

    file_out = f'{dir_out}/{id_run}.csv'
    if part is not None and nparts is not None:
        file_out = file_out.replace('.csv', f'-{part}-{nparts}.csv')

    df_res.write_csv(file_out, float_precision=4)


# EXAMPLES:
# ------------------------------------------------------------------------------
# for features for learning: find best weights on CV folds
"""
./tsfcst/find_best_weights.sh \
-t active \
-g feats \
-a 2022-07-01 \
-f no \
-m corner \
-n 0 \
-i active-cfips-20220701-naive-test-tld-1-0_0 \
-o no \
-k active-cfips-20220701-ema-test-tld-20-0_0 \
-d no \
-e active-cfips-20220701-theta-test-tld-50-0_02 \
-w no
"""

# for target for learning: find best weights on the test fold
"""
./tsfcst/find_best_weights.sh \
-t active \
-g target \
-a 2022-12-01 \
-f 2022-08-01 \
-m corner \
-n 0 \
-i active-cfips-20220701-naive-test-tld-1-0_0 \
-o no \
-k active-cfips-20220701-ema-test-tld-20-0_0 \
-d no \
-e active-cfips-20220701-theta-test-tld-50-0_02 \
-w no
"""

# for features for inference: find best weights on CV folds
"""
./tsfcst/find_best_weights.sh \
-t active \
-g feats \
-a 2022-12-01 \
-f no \
-m corner \
-n 0 \
-i active-cfips-20221201-naive-full-tld-1-0_0 \
-o no \
-k active-cfips-20221201-ema-full-tld-20-0_0 \
-d no \
-e active-cfips-20221201-theta-full-tld-50-0_02 \
-w no
"""
