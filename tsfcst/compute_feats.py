import os
from pathlib import Path
import logging
import argparse
from tqdm import tqdm
import json
import numpy as np
import pandas as pd
import polars as pl
import glob

import config
from etl import load_data, get_df_ts_by_cfips
from tsfcst.find_best_weights import load_best_weights
from tsfcst.time_series import TsData
from tsfcst.utils_tsfcst import get_feats, chowtest
from utils import set_display_options, describe_numeric
from tsfcst.models.inventory import MODELS


log = logging.getLogger(os.path.basename(__file__))
set_display_options()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--targetname', default='active', help='microbusiness_density, active')
    parser.add_argument('-a', '--asofdate', default='2022-12-01')
    args = parser.parse_args()
    return args


def get_id_run(targetname, asofdate: str, **kwargs):
    asofdate = asofdate.replace('-', '')
    return f"{targetname}-{asofdate}"


def load_feats(feats_name):
    return pl.read_csv(f'{config.DIR_ARTIFACTS}/compute_feats/{feats_name}/feats.csv')


if __name__ == '__main__':
    args = parse_args()

    # python -m tsfcst.compute_feats -t active -a 2022-12-01

    id_run = get_id_run(**vars(args))
    log.info(f'Running {os.path.basename(__file__)}, id_run={id_run}, with parameters: \n'
             + json.dumps(vars(args), indent=2))
    log.info('This computes time series features.')

    targetname = args.targetname
    asofdate = args.asofdate

    dir_out = f'{config.DIR_ARTIFACTS}/{Path(__file__).stem}/{id_run}'
    os.makedirs(dir_out, exist_ok=True)

    # load data
    df_train, df_test, df_census, df_pop = load_data()
    if asofdate is not None:
        df_train = df_train.filter(pl.col('first_day_of_month') <= pl.lit(asofdate).str.strptime(pl.Date, fmt='%Y-%m-%d'))
    list_cfips = sorted(np.unique(df_train['cfips']))
    n_cfips = len(list_cfips)
    log.debug(f'{len(list_cfips)} cfips loaded out of {n_cfips} available')

    list_feats = []
    for cfips in tqdm(list_cfips, unit='cfips', miniters=10, mininterval=3):
        df_ts = get_df_ts_by_cfips(cfips, targetname, df_train)
        ts = TsData(df_ts['first_day_of_month'], df_ts[targetname])

        feats = get_feats(df_ts[targetname])

        lastidx_20210101 = np.where(ts.time == pd.to_datetime('2021-01-01'))[0][0]
        chow_value, p_value = chowtest(ts.target, lastidx_20210101)

        feats = {**{'cfips': cfips}, **feats, **{'chow_val_20210101': chow_value, 'chow_pval_20210101': p_value}}
        list_feats.append(feats)

    df_feats = pl.from_records(list_feats)

    # join state
    df_states_enc = df_train.select(['state']).unique().sort('state') \
        .with_columns(pl.lit(1).alias('state_ordid')) \
        .with_columns(pl.cumsum('state_ordid'))
    df_cfips_states_enc = df_train.select(['cfips', 'state']).unique() \
        .join(df_states_enc, on='state') \
        .select(['cfips', 'state', 'state_ordid'])
    df_feats = df_feats.join(df_cfips_states_enc, on='cfips', how='left')

    # join census data
    df_census = df_census.with_columns([
        (pl.col('pct_bb_2021') / pl.col('pct_bb_2020').clip_min(0.00001) - 1).clip_min(-9.99).clip_max(9.99).alias('pct_bb_2021_yoy'),
        (pl.col('pct_college_2021') / pl.col('pct_college_2020').clip_min(0.00001) - 1).clip_min(-9.99).clip_max(9.99).alias('pct_college_2021_yoy'),
        (pl.col('pct_foreign_born_2021') / pl.col('pct_foreign_born_2020').clip_min(0.00001) - 1).clip_min(-9.99).clip_max(9.99).alias('pct_foreign_born_2021_yoy'),
        (pl.col('pct_it_workers_2021') / pl.col('pct_it_workers_2020').clip_min(0.00001) - 1).clip_min(-9.99).clip_max(9.99).alias('pct_it_workers_2021_yoy'),
        (pl.col('median_hh_inc_2021') / pl.col('median_hh_inc_2020').clip_min(0.00001) - 1).clip_min(-9.99).clip_max(9.99).alias('median_hh_inc_2021_yoy'),
    ])
    cols_census = [
        'cfips',
        'pct_bb_2021', 'pct_college_2021', 'pct_foreign_born_2021', 'pct_it_workers_2021', 'median_hh_inc_2021',
        'pct_bb_2021_yoy', 'pct_college_2021_yoy', 'pct_foreign_born_2021_yoy', 'pct_it_workers_2021_yoy', 'median_hh_inc_2021_yoy',
    ]
    df_census = df_census.select(cols_census)
    df_feats = df_feats.join(df_census, on='cfips', how='left')

    # join best theta of state
    map_aod_bestparams_theta_state = {
        '2022-07-01': 'active-state-20220701-theta-test-tld-100-0_02',
        '2022-12-01': 'active-state-20221201-theta-full-tld-100-0_02',
    }
    id_best_params_theta_state = map_aod_bestparams_theta_state.get(asofdate, None)
    assert id_best_params_theta_state is not None
    dir_best_params = f'{config.DIR_ARTIFACTS}/find_best_params/{id_best_params_theta_state}'
    files_best_params = sorted(glob.glob(f'{dir_best_params}/*.csv'))
    df_theta = pl.concat([pl.read_csv(f) for f in files_best_params]) \
        .select(['state', 'theta']) \
        .rename({'theta': 'theta_state'})
    df_feats = df_feats.join(df_theta, on='state', how='left')

    # join weights and cross-validation errors
    map_aod_id_weights_cv = {
        '2022-07-01': 'active-feats-naive_ema_theta-corner-20220701-no-manual_fix',
        '2022-12-01': 'active-feats-naive_ema_theta-corner-20221201-no-manual_fix',
    }
    id_weights_cv = map_aod_id_weights_cv.get(asofdate, None)
    assert id_weights_cv is not None
    df_weights_cv = load_best_weights(id_weights_cv)

    benchmark = 'naive'
    for m in MODELS.keys():
        if m not in df_weights_cv.columns or m == benchmark:
            continue
        df_weights_cv = df_weights_cv \
            .with_columns((pl.col(f'smape_{m}') - pl.col(f'smape_{benchmark}'))
                          .alias(f'diff_smape_{m}_{benchmark}'))

    map_weights_cv_names = {c: f'val_{c}' for c in df_weights_cv.columns if c != 'cfips'}
    df_weights_cv = df_weights_cv.rename(map_weights_cv_names).sort('cfips')
    df_feats = df_feats.join(df_weights_cv, on='cfips')

    df_feats.write_csv(f'{dir_out}/feats.csv', float_precision=4)

    log.debug(f'{len(df_feats.columns) - 1} features created')
    log.debug('\n' + str(df_feats.head()))
    log.debug('\n' + str(describe_numeric(df_feats.to_pandas())))
