import polars as pl
import glob

from utils import set_display_options, describe_numeric
import config


set_display_options()

weights_alias = 'active-full-naive_ema_theta-find_best_corner-20221201'
# weights_alias = 'active-naive_ema_theta-find_best_corner-20221201'

# params_map = {
#     'naive': 'active-20220701-naive-test-level-1-0_0',
#     'ma': 'active-20220701-ema-test-trend_level_damp-25-0_0',
#     'theta': 'active-20220701-theta-test-trend_level_damp-50-0_0'
# }

params_map = {
    'naive': 'active-20221201-naive-full-trend_level_damp-1-0_0',
    'ma': 'active-20221201-ma-full-trend_level_damp-20-0_0',
    'theta': 'active-20221201-theta-full-trend_level_damp-50-0_0'
}

file_weights = f'{config.DIR_ARTIFACTS}/find_best_weights/{weights_alias}/{weights_alias}.csv'
df_weights = pl.read_csv(file_weights)

naive_alias = params_map['naive']
dir_best_params = f'{config.DIR_ARTIFACTS}/find_best_params/{naive_alias}'
files_best_params = sorted(glob.glob(f'{dir_best_params}/*.csv'))
df_naive = pl.concat([pl.read_csv(f) for f in files_best_params])
df_naive = df_naive\
    .filter(pl.col('selected_trials') == 'best')\
    .select(['cfips', 'smape_cv_opt'])\
    .rename({'smape_cv_opt': 'smape_cv_opt_naive'})
print(df_naive.head().to_pandas())

theta_alias = params_map['theta']
dir_best_params = f'{config.DIR_ARTIFACTS}/find_best_params/{theta_alias}'
files_best_params = sorted(glob.glob(f'{dir_best_params}/*.csv'))
df_theta = pl.concat([pl.read_csv(f) for f in files_best_params])
df_theta = df_theta\
    .filter(pl.col('selected_trials') == 'best')\
    .select(['cfips', 'smape_cv_opt', 'theta'])\
    .rename({'smape_cv_opt': 'smape_cv_opt_theta', 'theta': 'par_theta'})
print(df_theta.head().to_pandas())

ma_alias = params_map['ma']
dir_best_params = f'{config.DIR_ARTIFACTS}/find_best_params/{ma_alias}'
files_best_params = sorted(glob.glob(f'{dir_best_params}/*.csv'))
df_ma = pl.concat([pl.read_csv(f) for f in files_best_params])
df_ma = df_ma \
    .filter(pl.col('selected_trials') == 'best') \
    .select(['cfips', 'smape_cv_opt', 'window']) \
    .rename({'smape_cv_opt': 'smape_cv_opt_ma', 'window': 'par_window'})
print(df_ma.head().to_pandas())

df = df_weights\
    .join(df_naive, on='cfips')\
    .join(df_ma, on='cfips')\
    .join(df_theta, on='cfips')

describe_numeric(df.to_pandas())

# replace theta with naive where it is equivalent
df = df.with_columns(
    ((pl.col('theta') == 1)
     & (pl.col('par_theta') <= 1.05)
     & ((pl.col('smape_naive') - pl.col('smape_theta')) <= 0.05)
     ).alias('theta_is_naive'))

print(df.filter(pl.col('theta_is_naive')).head().to_pandas())
df = df.with_columns([
    pl.when(pl.col('theta_is_naive')).then(0).otherwise(pl.col('theta')).alias('theta'),
    pl.when(pl.col('theta_is_naive')).then(1).otherwise(pl.col('naive')).alias('naive')
])
df = df.drop('theta_is_naive')

# replace ma with naive where it is equivalent
df = df.with_columns(
    ((pl.col('ma') == 1)
     & (pl.col('par_window') <= 1)
     & ((pl.col('smape_naive') - pl.col('smape_ma')) <= 0.05)
     ).alias('ma_is_naive'))
df.filter(pl.col('ma_is_naive')).to_pandas()

df = df.with_columns([
    pl.when(pl.col('ma_is_naive')).then(0).otherwise(pl.col('ma')).alias('ma'),
    pl.when(pl.col('ma_is_naive')).then(1).otherwise(pl.col('naive')).alias('naive')
])
df = df.drop('ma_is_naive')

print(describe_numeric(df.to_pandas()))

print(df.head().to_pandas())

df.write_csv(file_weights, float_precision=4)


# distribution by model
#      naive    ma  theta
# val:  0.25  0.25   0.50
# test: 0.60  0.10   0.30
# full: 0.48  0.11   0.41