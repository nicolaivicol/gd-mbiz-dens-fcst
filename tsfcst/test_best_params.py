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
from etl import load_data, get_df_ts_by_id
from tsfcst.time_series import TsData
from tsfcst.forecasters.forecaster import Forecaster, ForecasterConfig
from tsfcst.find_best_params import add_common_args, get_id_run


log = logging.getLogger(os.path.basename(__file__))


def parse_args():
    parser = argparse.ArgumentParser()
    parser = add_common_args(parser)
    parser.add_argument('-st', '--selected_trials', default='best', help='options: best, top10')
    parser.add_argument('-pt', '--periods_test', default=0, type=int,
                        help='use when forecasts on test are needed for evaluation and/pr growth rates')
    parser.add_argument('-po', '--periods_out', default=0, type=int,
                        help='use when growth rates are needed on out-of-sample, where no actual available')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    id_run = get_id_run(**vars(args))
    log.info(f'Running {os.path.basename(__file__)}, id_run={id_run}, with parameters: \n'
             + json.dumps(vars(args), indent=2))
    log.info('This tests best parameters found previously on validation and test.')

    target_name = args.targetname
    cv_args = config.CV_ARGS_DICT[args.cvargs]
    selected_trials = args.selected_trials
    idcol = args.idcol

    dir_out = f'{config.DIR_ARTIFACTS}/{Path(__file__).stem}/{id_run}'
    os.makedirs(dir_out, exist_ok=True)

    # load ts data
    df_train, df_test, df_census, df_pop = load_data()

    # identify IDs to run for
    list_ids = sorted(np.unique(df_train[idcol]))
    if idcol == 'state':
        list_ids.insert(0, 'all')
    log.debug(f'{len(list_ids)} {idcol}s loaded')

    # load best params (! it looks for the same run ID)
    dir_best_params = f'{config.DIR_ARTIFACTS}/find_best_params/{id_run}'
    files_best_params = sorted(glob.glob(f'{dir_best_params}/*.csv'))
    df_best_params = pl.concat([pl.read_csv(f) for f in files_best_params])
    df_best_params = df_best_params.filter(pl.col('selected_trials') == args.selected_trials)

    list_metrics_cv, list_df_fcsts_cv = [], []

    for id in tqdm(list_ids, unit=idcol, leave=False, mininterval=3):
        df_ts = get_df_ts_by_id(idcol, id, target_name, df_train)
        ts = TsData(df_ts['first_day_of_month'], df_ts[target_name])

        fcster_cfg = ForecasterConfig.from_df_with_best_params(id, df_best_params, idcol)
        fcster = Forecaster.from_config(data=ts, cfg=fcster_cfg)

        df_fcsts_cv, metrics_cv = fcster.cv(
            n_train_dates=cv_args['n_train_dates'],
            step_train_dates=cv_args['step_train_dates'],
            periods_val=cv_args['periods_val'],
            periods_test=args.periods_test,
            periods_out=args.periods_out,
            periods_val_last=cv_args['periods_val_last'],
        )

        metrics_cv = {**{idcol: id}, **vars(args), **metrics_cv}
        list_metrics_cv.append(metrics_cv)

        df_fcsts_cv[idcol] = id
        list_df_fcsts_cv.append(df_fcsts_cv)

    df_metrics_cv = pd.DataFrame.from_records(list_metrics_cv)
    df_fcsts_cv = pd.concat(list_df_fcsts_cv)

    df_metrics_cv.to_csv(f'{dir_out}/df_metrics_cv.csv', index=False)
    df_fcsts_cv.to_csv(f'{dir_out}/df_fcsts_cv.csv', index=False)

    log.debug('done.')


# EXAMPLES:
# ------------------------------------------------------------------------------
# for model, for features / testing/ growth rates:

# theta:
# python -m tsfcst.test_best_params -t active -m theta -c test -s tld -n 50 -r 0.02 -a 2022-07-01 -i cfips -pt 5 -po 0
# python -m tsfcst.test_best_params -t active -m theta -c full -s tld -n 50 -r 0.02 -a 2022-12-01 -i cfips -pt 0 -po 6

# naive:
# python -m tsfcst.test_best_params -t active -m naive -c full -s tld -n 1 -r 0 -a 2022-12-01 -i cfips -pt 0 -po 6

# ema:
# python -m tsfcst.test_best_params -t active -m ema -c full -s tld -n 20 -r 0 -a 2022-12-01 -i cfips -pt 0 -po 6

# by state, for growth rates:
# python -m tsfcst.test_best_params -t active -m theta -c test -s tld -n 100 -r 0.02 -a 2022-07-01 -i state -pt 5
# python -m tsfcst.test_best_params -t active -m theta -c full -s tld -n 100 -r 0.02 -a 2022-12-01 -i state -po 6

