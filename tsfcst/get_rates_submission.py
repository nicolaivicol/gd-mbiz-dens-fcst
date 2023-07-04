import os
import glob
import polars as pl
import pandas as pd

import config
from utils import set_display_options, describe_numeric
from etl import load_data

set_display_options()

asofdate = '2022-12-01'
asofdate_fcst = '2023-01-01'

id_fcst = 'best_public'
# id_fcst = 'ens-naive_ema_theta-20220701-active-wavg-test-weight-folds_5-active-20220701-active-target-naive_ema_theta-corner-20221201-20220801-manual_fix'
# id_fcst = 'ens-naive_ema_theta-20221201-active-wavg-full-weight-folds_1-active-20221201-active-target-naive_ema_theta-corner-20221201-20220801-manual_fix'
# id_fcst = 'combined'
dir_fcsts = f'{config.DIR_ARTIFACTS}/forecast_ensemble/{id_fcst}'

df_train, df_test, df_census, df_pop = load_data()
df_actual_last = df_train \
    .with_columns(pl.lit(asofdate).str.strptime(pl.Date, fmt='%Y-%m-%d').alias('asofdate')) \
    .filter(pl.col('first_day_of_month') == pl.lit(asofdate).str.strptime(pl.Date, fmt='%Y-%m-%d')) \
    .select(['cfips', 'asofdate', 'active'])

try:
    file_fcst = sorted(glob.glob(f'{dir_fcsts}/sub*.csv'))[-1]
    df_fcst = pl.read_csv(file_fcst)
    df_fcst = df_fcst \
        .with_columns(pl.col('row_id').str.split('_').alias('row_id')) \
        .with_columns(pl.col('row_id').arr.get(0).cast(pl.Int64).alias('cfips')) \
        .with_columns(pl.col('row_id').arr.get(1).str.strptime(pl.Date, fmt='%Y-%m-%d').alias('date'))\
        .filter(pl.col('date') >= pl.lit(asofdate_fcst).str.strptime(pl.Date, fmt='%Y-%m-%d'))
except:
    file_fcst = f'{dir_fcsts}/fcsts_all_models.csv'
    df_fcst = pl.read_csv(file_fcst)
    df_fcst = df_fcst \
        .select(['date', 'cfips', 'ensemble']) \
        .with_columns(pl.col('date').str.strptime(pl.Date, fmt='%Y-%m-%d').alias('date')) \
        .rename({'ensemble': 'microbusiness_density'}) \
        .filter(pl.col('date') >= pl.lit(asofdate_fcst).str.strptime(pl.Date, fmt='%Y-%m-%d'))

# convert to active
df_fcst = df_fcst \
    .join(df_pop.rename({'first_day_of_month': 'date'}), on=['cfips', 'date']) \
    .with_columns((pl.col('microbusiness_density') * pl.col('population') / 100).alias('forecast')) \
    .select(['cfips', 'date', 'forecast'])

df_rates = df_actual_last \
    .join(df_fcst, on='cfips') \
    .with_columns((pl.col('forecast') / pl.col('active') - 1).alias('rate_county')) \
    .select(['cfips', 'asofdate', 'date', 'rate_county'])

print(describe_numeric(
    df=df_rates.select(['rate_county']).to_pandas(),
    percentiles=[0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.95, 0.98, 0.99],
    stats_nans=False))

# rates by horizon/date
list_stats = []
for d in df_rates['date'].unique():
    stats = describe_numeric(df_rates.filter(pl.col('date') == d).select(['rate_county']).to_pandas(), stats_nans=False)
    stats['date'] = d
    list_stats.append(stats)
stats = pd.concat(list_stats)
print(stats)

print((stats[['date', 'mean']]))

print(pl.from_pandas(stats[['date', 'mean']]))

# ┌────────────┬──────────┐
# │ date       ┆ mean     │
# ╞════════════╪══════════╡
# │ 2023-01-01 ┆ 0.003256 │
# │ 2023-02-01 ┆ 0.005212 │
# │ 2023-03-01 ┆ 0.006449 │
# │ 2023-04-01 ┆ 0.007289 │
# │ 2023-05-01 ┆ 0.00794  │
# │ 2023-06-01 ┆ 0.008466 │
# └────────────┴──────────┘

# ┌────────────┬──────────┐
# │ date       ┆ mean     │
# ╞════════════╪══════════╡
# │ 2023-01-01 ┆ 0.003256 │
# │ 2023-02-01 ┆ 0.003946 │
# │ 2023-03-01 ┆ 0.004767 │
# │ 2023-04-01 ┆ 0.00554  │
# │ 2023-05-01 ┆ 0.006318 │
# │ 2023-06-01 ┆ 0.007104 │
# └────────────┴──────────┘

# stats for the best public submission:
#                count  mean   std    min     1%    5%   10%   25%   50%   75%   95%   98%   99%   max
# rate_county 3135.000 0.003 0.002 -0.000 -0.000 0.000 0.000 0.000 0.004 0.005 0.006 0.006 0.007 0.007

# ens-naive_ema_theta-20221201-active-wavg-full-weight-folds_1-active-20221201-active-target-naive_ema_theta-corner-20221201-20220801-manual_fix
#                count  mean   std    min     1%     5%    10%    25%   50%   75%   95%   98%   99%   max
# rate_county 3135.000 0.001 0.003 -0.016 -0.006 -0.002 -0.000 -0.000 0.000 0.001 0.004 0.007 0.009 0.048

# ens-naive_theta-20221201-active-max-no:
#                count  mean   std    min     1%     5%    10%   25%   50%   75%   95%   98%   99%   max
# rate_county 3135.000 0.002 0.016 -0.000 -0.000 -0.000 -0.000 0.000 0.000 0.002 0.008 0.012 0.017 0.654

# ens-naive_ema_theta-20220701-active-wavg-test-weight-folds_5-active-20220701-active-target-naive_ema_theta-corner-20221201-20220801-manual_fix
#                count   mean   std    min     1%     5%    10%    25%   50%   75%   95%   98%   99%   max
# rate_county 3134.000 -0.000 0.004 -0.059 -0.012 -0.006 -0.003 -0.000 0.000 0.001 0.004 0.006 0.009 0.045

dir_out = f'{config.DIR_ARTIFACTS}/get_rates_submission'
os.makedirs(dir_out, exist_ok=True)
file_out = f'{dir_out}/{id_fcst}.csv'
df_rates.write_csv(file_out, float_precision=6)

print(f'rates saved to: {file_out}')
