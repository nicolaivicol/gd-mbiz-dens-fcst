import os
from pathlib import Path
import logging
import argparse
from tqdm import tqdm
import json
import numpy as np
import polars as pl

import config
from etl import load_data
from tsfcst.weights_finder import WeightsFinder


log = logging.getLogger(os.path.basename(__file__))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--target_name', default='microbusiness_density', help='microbusiness_density, active')
    parser.add_argument('-g', '--tag', default='test', help='options: cv, test, full')
    parser.add_argument('-a', '--asofdate', default='2022-10-01')
    parser.add_argument('-f', '--fromdate', default='2022-08-01')
    parser.add_argument('-n', '--ntrials', default=200, type=int)
    parser.add_argument('-i', '--naive', default='microbusiness_density-20220701-naive-test-trend_level_damp-1-0_0')
    parser.add_argument('-o', '--ma', default='microbusiness_density-20220701-ma-test-trend_level_damp-100-0_0')
    parser.add_argument('-e', '--theta', default='microbusiness_density-20220701-theta-test-trend_level_damp-100-0_0')
    parser.add_argument('-w', '--hw', default='microbusiness_density-20220701-hw-test-trend_level_damp-100-0_0')
    parser.add_argument('-p', '--part', type=int)
    parser.add_argument('-x', '--nparts', type=int)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    # python -m tsfcst.find_best_weights -t microbusiness_density -g test -a 2022-10-01 -f 2022-08-01 -n 200 -i "microbusiness_density-20220701-naive-test-trend_level_damp-1-0_0" -o "microbusiness_density-20220701-ma-test-trend_level_damp-100-0_0" -e "microbusiness_density-20220701-theta-test-trend_level_damp-100-0_0" -w "microbusiness_density-20220701-hw-test-trend_level_damp-100-0_0"

    log.info(f'Running {os.path.basename(__file__)} with parameters: \n'
             + json.dumps(vars(args), indent=2))
    log.info('This finds best weights to apply when ensembling the models: naive, ma, theta, hw.')

    target_name = args.target_name
    tag = args.tag
    asofdate = args.asofdate
    fromdate = args.fromdate
    ntrials = args.ntrials
    part = args.part
    nparts = args.nparts
    dict_models_id_run = {'naive': args.naive, 'ma': args.ma, 'theta': args.theta, 'hw': args.hw}
    model_names = list(dict_models_id_run.keys())

    id_run = f"{target_name}-{tag}-{asofdate.replace('-', '')}"
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

    if fromdate is not None and fromdate.lower() not in ['none', 'null', 'na', '', 'nan', 'no']:
        df_fcsts = df_fcsts.filter(pl.col('date') >= pl.lit(fromdate).str.strptime(pl.Date, fmt='%Y-%m-%d'))

    log.debug('count of dates: \n' + str(df_fcsts.select(['cfips', 'date']).groupby(['cfips']).count().describe()))

    df_fcsts = df_fcsts.with_columns(pl.col('cfips').cast(pl.Int32))

    # load actual data
    df_actual, _, _ = load_data()
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

        res = WeightsFinder.find_best(n_trials=ntrials)

        msg_str = ' | '.join([f"{name_}={w:.2f}" for name_, w in res['best_params'].items()])
        log.debug(f'best weights: ' + msg_str)

        smapes_ = [WeightsFinder.smape(w) for w in ([res['weights']] + WeightsFinder.params_corners())]
        msg_str = ' | '.join([f'{name_}={smape_:.2f}' for name_, smape_ in zip(['best'] + model_names, smapes_)])
        log.debug(f'compare SMAPEs:  ' + msg_str)

        res_dict = {
            **{'cfips': cfips, 'smape': np.round(res['smape'], 4)},
            **res['best_params'],
            **{'weights_arr': json.dumps(res['weights']),
               'weights_dict': json.dumps(res['best_params'])},
        }
        list_res.append(res_dict)

    df_res = pl.DataFrame(list_res)

    file_out = f'{dir_out}/{id_run}.csv'
    if part is not None and nparts is not None:
        file_out = file_out.replace('.csv', f'-{part}-{nparts}.csv')

    df_res.write_csv(file_out, float_precision=4)
