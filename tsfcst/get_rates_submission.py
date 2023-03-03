import os
import glob
import polars as pl

import config
from utils import set_display_options, describe_numeric
from etl import load_data

set_display_options()

asofdate = '2022-12-01'
asofdate_fcst = '2023-01-01'

# load submission
if len(glob.glob(f'{dir_fcsts}/sub*.csv')) == 1:
    file_submission = glob.glob(f'{dir_fcsts}/sub*.csv')
    df_fcsts = pl.read_csv(file_submission[0]).rename({'microbusiness_density': 'submission'})
    df_fcsts = df_fcsts \
        .with_columns(pl.col('row_id').str.split('_').alias('row_id')) \
        .with_columns(pl.col('row_id').arr.get(0).cast(pl.Int64).alias('cfips')) \
        .with_columns(pl.col('row_id').arr.get(1).str.strptime(pl.Date, fmt='%Y-%m-%d').alias('date'))
    df_fcsts = df_fcsts.join(df_pop.rename({'first_day_of_month': 'date'}), on=['cfips', 'date']) \
        .select(['cfips', 'date', 'submission', 'population']) \
        .sort(['cfips', 'date'])

df_train, df_test, df_census, df_pop = load_data()
df_actual_last = df_train \
    .with_columns(pl.lit(asofdate).str.strptime(pl.Date, fmt='%Y-%m-%d').alias('asofdate')) \
    .filter(pl.col('first_day_of_month') == pl.lit(asofdate).str.strptime(pl.Date, fmt='%Y-%m-%d')) \
    .select(['cfips', 'asofdate', 'active'])

df_fcst = pl.read_csv(f'{config.DIR_DATA}/best_public_submission.csv')
df_fcst_first = df_fcst \
    .with_columns(pl.col('row_id').str.split('_').alias('row_id')) \
    .with_columns(pl.col('row_id').arr.get(0).cast(pl.Int64).alias('cfips')) \
    .with_columns(pl.col('row_id').arr.get(1).str.strptime(pl.Date, fmt='%Y-%m-%d').alias('date'))\
    .filter(pl.col('date') == pl.lit(asofdate_fcst).str.strptime(pl.Date, fmt='%Y-%m-%d'))

df_fcst_first = df_fcst_first.join(df_pop.rename({'first_day_of_month': 'date'}), on=['cfips', 'date']) \
    .with_columns((pl.col('microbusiness_density') * pl.col('population') / 100).alias('forecast')) \
    .select(['cfips', 'date', 'forecast'])

df_rates = df_actual_last.join(df_fcst_first, on='cfips') \
    .with_columns((pl.col('forecast') / pl.col('active') - 1).alias('rate_county')) \
    .select(['cfips', 'asofdate', 'date', 'rate_county'])

print(describe_numeric(
    df=df_rates.select(['rate_county']).to_pandas(),
    percentiles=[0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.95, 0.98, 0.99],
    stats_nans=False))

# stats for the best public submission:
#                count  mean   std    min     1%     5%    10%   25%   50%   75%   95%   98%   99%   max
# rate_county 3135.000 0.002 0.002 -0.003 -0.000 -0.000 -0.000 0.001 0.003 0.004 0.005 0.005 0.005 0.006

dir_out = f'{config.DIR_ARTIFACTS}/get_rates_submission'
os.makedirs(dir_out, exist_ok=True)
file_out = f'{dir_out}/rates-best-public-submission.csv'
df_rates.write_csv(file_out, float_precision=5)

print(f'rates saved to: {file_out}')
