import os
import glob
import polars as pl
import pandas as pd

import config
from utils import set_display_options, describe_numeric
from etl import load_data

set_display_options()

asofdate = '2022-07-01'

df_train, df_test, df_census, df_pop = load_data()

df_last = df_train \
    .with_columns(pl.lit(asofdate).str.strptime(pl.Date, fmt='%Y-%m-%d').alias('asofdate')) \
    .filter(pl.col('first_day_of_month') == pl.lit(asofdate).str.strptime(pl.Date, fmt='%Y-%m-%d')) \
    .rename({'active': 'last'}) \
    .select(['cfips', 'asofdate', 'last'])

df_next = df_train \
    .rename({'active': 'next'}) \
    .select(['cfips', 'first_day_of_month', 'next'])

df_rate = df_last.join(df_next, on='cfips') \
    .filter(pl.col('first_day_of_month') > pl.col('asofdate')) \
    .sort(['cfips', 'first_day_of_month']) \
    .rename({'first_day_of_month': 'date'}) \
    .with_columns((pl.col('date').cumcount().over('cfips') + 1).alias('horizon')) \
    .with_columns((pl.col('next') / pl.col('last') - 1).clip_min(-1).clip_max(1).alias('rate'))\
    .with_columns(pl.when(pl.col('rate').is_nan()).then(0).otherwise(pl.col('rate')).alias('rate'))\
    .select(['cfips', 'asofdate', 'date', 'horizon', 'rate'])

list_stats = []
for h in sorted(list(set(df_rate['horizon']))):
    stats = describe_numeric(
        df=df_rate.filter(pl.col('horizon') == h).select(['rate']).to_pandas(),
        percentiles=[0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95],
        stats_nans=False
    )
    stats['horizon'] = h
    list_stats.append(stats)
stats = pd.concat(list_stats)
print(stats)

dir_out = f'{config.DIR_ARTIFACTS}/get_rates_target'
os.makedirs(dir_out, exist_ok=True)
file_out = f"{dir_out}/rates_target_{asofdate.replace('-', '')}.csv"
df_rate.write_csv(file_out, float_precision=5)

print(df_rate.head())

print(f'rates saved to: {file_out}')

# ┌───────┬────────────┬────────────┬─────────┬───────────┐
# │ cfips ┆ asofdate   ┆ date       ┆ horizon ┆ rate      │
# ╞═══════╪════════════╪════════════╪═════════╪═══════════╡
# │ 1001  ┆ 2022-07-01 ┆ 2022-08-01 ┆ 1       ┆ -0.004107 │
# │ 1001  ┆ 2022-07-01 ┆ 2022-09-01 ┆ 2       ┆ 0.001369  │
# │ 1001  ┆ 2022-07-01 ┆ 2022-10-01 ┆ 3       ┆ 0.007529  │
# │ 1001  ┆ 2022-07-01 ┆ 2022-11-01 ┆ 4       ┆ 0.001369  │
# │ 1001  ┆ 2022-07-01 ┆ 2022-12-01 ┆ 5       ┆ 0.009582  │
# └───────┴────────────┴────────────┴─────────┴───────────┘

#         count   mean   std    min     5%    10%    25%    50%   75%   90%   95%   max  horizon
# rate 3135.000 -0.006 0.026 -0.302 -0.034 -0.025 -0.012 -0.005 0.000 0.010 0.021 0.602        1
# rate 3135.000 -0.002 0.045 -0.822 -0.045 -0.029 -0.012 -0.002 0.006 0.021 0.039 1.000        2
# rate 3135.000  0.001 0.058 -0.822 -0.050 -0.032 -0.013  0.000 0.011 0.031 0.052 1.000        3
# rate 3135.000  0.003 0.065 -0.822 -0.060 -0.036 -0.014  0.000 0.014 0.038 0.062 1.000        4
# rate 3135.000  0.012 0.089 -0.822 -0.064 -0.037 -0.011  0.005 0.023 0.058 0.091 1.000        5
