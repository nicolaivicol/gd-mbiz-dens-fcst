import numpy as np
import polars as pl
import logging
import os

import config
from utils import describe_numeric, set_display_options
from etl import load_data
from tsfcst.utils_tsfcst import smape

log = logging.getLogger(os.path.basename(__file__))


set_display_options()

df_train, _, _, _ = load_data()

dir_fcsts = f'{config.DIR_ARTIFACTS}/forecast_ensemble'

fcsts_to_compare = {
    'test-w': 'active-weight-test-active-naive_ema_theta-find_best_corner-20221201-20220701',
    'naive': 'active-naive-naive-20220701',
    'sma': 'active-sma-ma-20220701',
    'ema': 'active-ema-ma-20220701',
    'drift': 'active-drift-drift-20220701',
    'driftr-ws_1': 'model-driftr-wstate_1-driftr-20220701-active-single-driftr',
    'driftr-wct_1': 'model-driftr-wcountry_1-driftr-20220701-active-single-driftr',
    'driftr-ws_050-wct_050': 'active-drift-wavg-drift-w_s-050-w_ct_50-20220701',
    'driftr-s050_wct050_m150': 'model-driftr-s050_wct050_m150-driftr-20220701-active-single-driftr',
    'driftr-pred-r': 'model-driftr-lgbm-driftr-20220701-active-single-driftr',
    'theta': 'active-theta-v2-theta-20220701',
    'theta-reg-0.02': 'active-theta-r0_02-weighted_average-theta-20220701',
    'theta-reg-0.25': 'active-theta-025-theta-20220701',
    'theta-reg-1': 'active-theta-reg-1-theta-20220701',
    'hw': 'active-hw-hw-20220701',
    'hw-050': 'active-hw-050-hw-20220701',
    'hw-2': 'active-hw-2_0-hw-20220701',
    'hw-level': 'active-hw-level-hw-20220701',
    'avg_naive_theta': 'active-avg_naive_theta-avg_naive_theta-20220701',
    'cv-w': 'active-weight-cv-active-naive_ema_theta-find_best_corner-20220701-20220701',
    'model-w': 'active-lgbm-naive-ma-theta-folds_5-active-20220701-active-naive_ema_theta-find_best_corner-20221201-20220701',
    'model-sq-w': 'active-lgbm-bin-sq-naive-ma-theta-folds_5-active-20220701-active-naive_ema_theta-find_best_corner-20221201-20220701',
    'lgbm-bin-naive-ma-h025-theta-h025': 'active-lgbm-bin-naive-ma-h025-theta-h025-lgbm-bin-naive-ma-h025-theta-h025-folds_5-active-20220701-active-naive_ema_theta-find_best_corner-20221201-20220701',
    'lgbm-bin-naive-theta-h025': 'active-ens-lgbm-bin-naive-theta-h025-folds_5-active-20220701-active-naive_ema_theta-find_best_corner-20221201-20220701',
    'lgbm-bin-naive-theta-h035': 'active-ens-lgbm-bin-naive-theta-h035-folds_5-active-20220701-active-naive_ema_theta-find_best_corner-20221201-20220701',
    'lgbm-bin-naive-theta-h040': 'active-ens-lgbm-bin-naive-theta-h040-folds_5-active-20220701-active-naive_ema_theta-find_best_corner-20221201-20220701',
    'lgbm-bin-naive-ma-h040-theta-h040': 'active-ens-lgbm-bin-naive-ma-h040-theta-h040-folds_5-active-20220701-active-naive_ema_theta-find_best_corner-20221201-20220701',
    'median-naive-ma-theta': 'active-ens-median-lgbm-bin-naive-theta-h040-folds_1-active-20220701-active-test-naive_ema_theta-find_best_corner-20221201-20220701',
    'max-naive-ma-theta': 'active-ens-maximum-lgbm-bin-naive-theta-h040-folds_1-active-20220701-active-test-naive_ema_theta-find_best_corner-20221201-20220701',
    'lgbm-naive-ema-theta-overriden': 'ens-naive_ema_theta-20220701-active-wavg-test-weight-folds_5-active-20220701-active-target-naive_ema_theta-corner-20221201-20220801-manual_fix',
}

df_smapes = None

for name_fcst, run_id in fcsts_to_compare.items():
    df_fcsts = pl.read_csv(f'{dir_fcsts}/{run_id}/fcsts_all_models.csv', parse_dates=True)

    df_fcst = df_train \
        .select(['cfips', 'first_day_of_month', 'microbusiness_density']) \
        .rename({'first_day_of_month': 'date', 'microbusiness_density': 'actual'}) \
        .join(df_fcsts.select(['cfips', 'date', 'ensemble']), on=['cfips', 'date'])

    list_smape = []
    for cfips in sorted(list(np.unique(df_fcst['cfips']))):
        tmp = df_fcst.filter(pl.col('cfips') == cfips)
        smape_ = smape(tmp['actual'], tmp['ensemble'])
        list_smape.append({'cfips': cfips, name_fcst: smape_})

    df_smape = pl.DataFrame(list_smape)

    if df_smapes is None:
        df_smapes = df_smape
    else:
        df_smapes = df_smapes.join(df_smape, on='cfips')

#
# setups_to_compare = {
#     'naive': 'microbusiness_density-20220701-naive-test-trend_level_damp-1-0_0',
#     'ma': 'microbusiness_density-20220701-ma-test-trend_level_damp-100-0_0',
#     'theta': 'microbusiness_density-20220701-theta-test-trend_level_damp-100-0_0',
#     'hw': 'microbusiness_density-20220701-hw-test-trend_level_damp-100-0_0',
# }
#
# dir_test_best_params = f'{config.DIR_ARTIFACTS}/test_best_params'
#
# df = None
# cols = ['cfips', 'smape_avg_val', 'smape_avg_test']
#
# for alias_, run_id in setups_to_compare.items():
#     file_ = f'{dir_test_best_params}/{run_id}/df_metrics_cv.csv'
#     df_alias = pl.read_csv(file_)\
#         .select(cols)\
#         .rename({'smape_avg_val': f'smape_avg_val_{alias_}',
#                  'smape_avg_test': f'smape_avg_test_{alias_}'})
#
#     if df is None:
#         df = df_alias
#     else:
#         df = df.join(df_alias, on='cfips')
#
# df = df.join(df_smapes, on='cfips')

df = df_smapes

summary_ = describe_numeric(df.to_pandas(), stats_nans=False)

log.info('\n' + str(summary_))

# LB (before release of new data):
# weights from cv give 2.4887
# weights from last 3 periods gives 1.5172
# weights to sum-up to 1 give 1.4278
# weights as 1/errors^2 on test (target=active) 1.449
# naive gives 1.0939

# LB (after the release of new data)
# 1.5924 - ens, max (naive, ma, theta)
# 1.4633 - naive
# 1.4544 - ens, naive + theta (h>0.40), weights by lgbm
# 1.422  - ens, driftr(wc-050 wct-050) + theta (h>0.40), weights by lgbm
# 1.42   - driftr(wc-050 wct-050)

#                                      count      mean       std      min       5%       25%       50%       75%       95%       98%       99%       max
# cfips                             3135.000 30376.038 15145.863 1001.000 5096.400 18178.000 29173.000 45076.000 53063.600 55063.640 55124.320 56045.000
# naive                             3135.000     2.442     4.431    0.000    0.318     0.735     1.356     2.655     7.376    11.996    18.889   112.475
# sma                               3135.000     2.945     6.123    0.000    0.353     0.799     1.541     3.142     8.749    14.637    24.604   176.432
# ema                               3135.000     2.906     5.915    0.000    0.339     0.797     1.516     3.111     8.803    14.869    22.739   171.487
# drift                             3135.000     2.621     4.556    0.000    0.332     0.784     1.499     2.906     7.778    12.316    19.198   116.907
# theta                             3135.000     2.682     4.761    0.000    0.342     0.796     1.514     2.939     7.661    12.783    19.347   115.187
# theta-reg-0.02                    3135.000     2.655     4.810    0.000    0.330     0.770     1.474     2.894     7.581    12.903    19.493   115.188
# theta-reg-0.25                    3135.000     2.648     5.910    0.000    0.330     0.738     1.437     2.797     7.420    13.136    19.493   200.000
# theta-reg-1                       3135.000     2.596     5.797    0.000    0.331     0.736     1.405     2.749     7.482    12.577    19.494   200.000
# hw                                3135.000     4.345     6.308    0.000    0.494     1.344     2.661     4.980    13.004    21.129    27.362   104.368
# hw-050                            3135.000     3.412     6.115    0.000    0.381     1.034     1.969     3.825    10.021    15.964    24.192   151.251
# hw-2                              3135.000     3.648     6.178    0.000    0.460     1.205     2.229     4.056    10.136    16.395    25.013   151.251
# hw-level                          3135.000     3.035     6.017    0.000    0.345     0.823     1.566     3.218     9.003    15.933    24.988   138.691
# avg_naive_theta                   3135.000     2.534     4.648    0.000    0.320     0.746     1.412     2.700     7.512    12.140    19.968   114.003
# cv-w                              3135.000     2.707     4.885    0.000    0.333     0.767     1.507     3.039     7.970    13.294    19.458   114.678
# test-w                            3135.000     2.239     4.281    0.000    0.289     0.633     1.191     2.423     6.788    11.465    17.112   112.475
# model-w                           3135.000     2.559     5.908    0.000    0.318     0.725     1.381     2.730     7.555    12.139    18.979   200.000
# model-sq-w                        3135.000     2.502     5.650    0.000    0.323     0.724     1.361     2.678     7.317    12.083    18.907   200.000
# lgbm-bin-naive-ma-h025-theta-h025 3135.000     2.463     4.477    0.000    0.317     0.725     1.363     2.705     7.420    12.131    18.573   113.555
# lgbm-bin-naive-theta-h025         3135.000     2.453     4.479    0.000    0.314     0.717     1.357     2.669     7.381    12.197    18.838   113.555
# lgbm-bin-naive-theta-h035         3135.000     2.438     4.441    0.000    0.314     0.709     1.353     2.671     7.326    11.968    18.889   113.555
# lgbm-bin-naive-theta-h040         3135.000     2.434     4.437    0.000    0.314     0.723     1.350     2.655     7.376    11.996    18.889   113.555
# lgbm-bin-naive-ma-h040-theta-h040 3135.000     2.435     4.438    0.000    0.314     0.723     1.347     2.647     7.376    11.996    18.889   113.555
# median-naive-ma-theta             3135.000     2.481     4.566    0.000    0.333     0.739     1.387     2.690     7.293    11.996    18.933   112.475
# max-naive-ma-theta                3135.000     2.738     5.582    0.000    0.333     0.787     1.487     2.955     7.863    13.420    21.545   171.487
# driftr                            3135.000     2.565     4.422    0.082    0.370     0.834     1.507     2.830     7.385    12.181    18.834   112.817