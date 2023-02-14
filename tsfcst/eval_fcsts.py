import polars as pl
import logging
import os

import config
from utils import describe_numeric, set_display_options

log = logging.getLogger(os.path.basename(__file__))


set_display_options()

setups_to_compare = {
    'naive': 'microbusiness_density-20220701-naive-test-trend_level_damp-1-0_0',
    'ma': 'microbusiness_density-20220701-ma-test-trend_level_damp-100-0_0',
    'theta': 'microbusiness_density-20220701-theta-test-trend_level_damp-100-0_0',
    'hw': 'microbusiness_density-20220701-hw-test-trend_level_damp-100-0_0',
}

dir_test_best_params = f'{config.DIR_ARTIFACTS}/test_best_params'

df = None
cols = ['cfips', 'smape_avg_val', 'smape_avg_test']

for alias_, run_id in setups_to_compare.items():
    file_ = f'{dir_test_best_params}/{run_id}/df_metrics_cv.csv'
    df_alias = pl.read_csv(file_)\
        .select(cols)\
        .rename({'smape_avg_val': f'smape_avg_val_{alias_}',
                 'smape_avg_test': f'smape_avg_test_{alias_}'})

    if df is None:
        df = df_alias
    else:
        df = df.join(df_alias, on='cfips')

summary_ = describe_numeric(df.to_pandas(), stats_nans=False)

log.info('\n' + str(summary_))

#                         count      mean       std      min       5%       25%       50%       75%       95%       max
# cfips                3135.000 30376.038 15145.863 1001.000 5096.400 18178.000 29173.000 45076.000 53063.600 56045.000
# smape_avg_val_naive  3135.000     3.151     4.233    0.000    0.735     1.322     2.056     3.563     8.449    95.165
# smape_avg_test_naive 3135.000     1.873     3.475    0.000    0.196     0.542     1.027     2.079     5.732    94.376
# smape_avg_val_ma     3135.000     2.960     5.195    0.000    0.575     1.064     1.752     3.096     8.030   135.731
# smape_avg_test_ma    3135.000     3.193     6.425    0.000    0.288     0.798     1.731     3.393    10.438   170.528
# smape_avg_val_theta  3135.000     2.730     4.862    0.000    0.597     1.032     1.629     2.873     7.616   143.092
# smape_avg_test_theta 3135.000     2.169     4.836    0.000    0.244     0.609     1.190     2.341     6.245   155.340
# smape_avg_val_hw     3135.000     3.324     6.235    0.000    0.605     1.104     1.849     3.459     9.623   165.440
# smape_avg_test_hw    3135.000     3.599     6.532    0.000    0.317     0.989     1.986     3.963    11.278   163.856
