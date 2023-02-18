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
from etl import load_data, get_df_ts_by_cfips
from tsfcst.time_series import TsData
from tsfcst.utils_tsfcst import get_feats, chowtest
from utils import set_display_options, describe_numeric


log = logging.getLogger(os.path.basename(__file__))
set_display_options()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--targetname', default='microbusiness_density', help='microbusiness_density, active')
    parser.add_argument('-a', '--asofdate', default='2022-07-01')
    args = parser.parse_args()
    return args


def get_id_run(targetname, asofdate: str, **kwargs):
    asofdate = asofdate.replace('-', '')
    return f"{targetname}-{asofdate}"


if __name__ == '__main__':
    args = parse_args()

    # python -m tsfcst.compute_feats -t microbusiness_density -a 2022-07-01

    id_run = get_id_run(**vars(args))
    log.info(f'Running {os.path.basename(__file__)}, id_run={id_run}, with parameters: \n'
             + json.dumps(vars(args), indent=2))
    log.info('This computes time series features.')

    targetname = args.targetname
    asofdate = args.asofdate

    dir_out = f'{config.DIR_ARTIFACTS}/{Path(__file__).stem}/{id_run}'
    os.makedirs(dir_out, exist_ok=True)

    # load data
    df_train, _, _ = load_data()
    if asofdate is not None:
        df_train = df_train.filter(pl.col('first_day_of_month') <= pl.lit(asofdate).str.strptime(pl.Date, fmt='%Y-%m-%d'))
    list_cfips = sorted(np.unique(df_train['cfips']))
    n_cfips = len(list_cfips)
    log.debug(f'{len(list_cfips)} cfips loaded out of {n_cfips} available')

    list_feats = []
    for cfips in tqdm(list_cfips, unit='cfips', miniters=10, mininterval=3):
        # log.debug(f'cfips={cfips}')
        df_ts = get_df_ts_by_cfips(cfips, targetname, df_train)
        ts = TsData(df_ts['first_day_of_month'], df_ts[targetname])
        feats = get_feats(df_ts[targetname])
        lastidx_20210101 = np.where(ts.time == pd.to_datetime('2021-01-01'))[0][0]  # ts.time.index(cfips)
        chow_value, p_value = chowtest(ts.target, lastidx_20210101)
        feats = {**{'cfips': cfips}, **feats, **{'chow_val_20210101': chow_value, 'chow_pval_20210101': p_value}}
        list_feats.append(feats)

    df_feats = pl.from_records(list_feats)
    df_feats.write_csv(f'{dir_out}/feats.csv', float_precision=4)

    df_feats = df_feats.to_pandas()
    log.debug('\n' + str(describe_numeric(df_feats)))
    log.debug('\n' + str(df_feats.head()))

