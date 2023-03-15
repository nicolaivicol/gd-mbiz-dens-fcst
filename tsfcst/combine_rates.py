import polars as pl
import pandas as pd
import os

import config
from utils import describe_numeric, set_display_options

set_display_options()

id_pub = 'best_public'
id_ens = 'ens-naive_ema_theta-20221201-active-wavg-full-weight-folds_1-active-20221201-active-target-naive_ema_theta-corner-20221201-20220801-manual_fix'

file_pub = f'{config.DIR_ARTIFACTS}/get_rates_submission/{id_pub}.csv'
df_rates_pub = pl.read_csv(file_pub, try_parse_dates=True)

file_ens = f'{config.DIR_ARTIFACTS}/get_rates_submission/{id_ens}.csv'
df_rates_ens = pl.read_csv(file_ens, try_parse_dates=True)

df_rates_pub_1st = df_rates_pub \
    .filter(pl.col('date') == pl.lit('2023-01-01').str.strptime(pl.Date, fmt='%Y-%m-%d')) \
    .rename({'rate_county': 'rate_county_pub'}) \
    .select(['cfips', 'rate_county_pub'])

df_rates_ens_1st = df_rates_ens \
    .filter(pl.col('date') == pl.lit('2023-01-01').str.strptime(pl.Date, fmt='%Y-%m-%d')) \
    .rename({'rate_county': 'rate_county_ens'}) \
    .select(['cfips', 'rate_county_ens'])

df_rates_ens = df_rates_ens \
    .join(df_rates_ens_1st, on='cfips') \
    .join(df_rates_pub_1st, on='cfips')

# adjust to higher rates where rate_county_ens > rate_county_pub
# e.g.: 48415, 48053, 12123, 48199
# df_rates_ens.filter(pl.col('cfips') == 1001)
# df_rates_ens.filter(pl.col('cfips') == 48415)
# df_rates_ens = df_rates_ens \
#     .with_columns(pl.when(pl.col('rate_county_ens') > pl.col('rate_county_pub'))
#                   .then(0.5 * pl.col('rate_county_ens') + 0.5 * pl.col('rate_county_pub'))
#                   .otherwise(pl.col('rate_county_pub'))
#                   .alias('rate_county_pub'))


# decrease my rates when greater than best public
print(df_rates_ens.filter((pl.col('rate_county_ens') > pl.col('rate_county_pub'))
                          & (pl.col('date').cast(str) == '2023-01-01')))
df_rates_ens = df_rates_ens \
    .with_columns(
        pl.when(pl.col('rate_county_ens') > pl.col('rate_county_pub'))
        .then((0.5 * pl.col('rate_county_ens') + 0.5 * pl.col('rate_county_pub')) / (pl.col('rate_county_ens')))
        .otherwise(1.0)
        .alias('adj_to_public'))
print(describe_numeric(df_rates_ens.select(['adj_to_public']).to_pandas(),
                       percentiles=[0.05, 0.10, 0.15, 0.20, 0.25, 0.50, 0.75, 0.95]))
df_rates_ens = df_rates_ens \
    .with_columns(pl.col('rate_county') * pl.col('adj_to_public')) \
    .with_columns(pl.col('rate_county_ens') * pl.col('adj_to_public'))\
    .drop('adj_to_public')

# replace growth rate for 2023-01-01 with best public
df_rates_ens = df_rates_ens \
    .with_columns((pl.col('rate_county') - pl.col('rate_county_ens') + pl.col('rate_county_pub')).clip_min(0).alias('rate')) \
    .with_columns(pl.when(pl.col('rate') < pl.col('rate_county_pub')).then(pl.col('rate_county_pub')).otherwise(pl.col('rate')).alias('rate'))

# propagate some of the best public rate forward when it is higher than my rate
print(df_rates_ens.filter((pl.col('rate_county_ens') < pl.col('rate_county_pub'))
                          & (pl.col('date').cast(str) == '2023-02-01')))
df_rates_ens = df_rates_ens.with_columns((pl.col('rate_county_pub').clip_min(0) - pl.col('rate_county_ens').clip_min(0)).clip_min(0).alias('diff'))
df_adj_factor = pl.DataFrame({
    'date': ['2023-01-01', '2023-02-01', '2023-03-01', '2023-04-01', '2023-05-01', '2023-06-01'],
    'adj_factor': [0, 0.50, 0.75, 0.875, 0.925, 0.95]}
).with_columns(pl.col('date').str.strptime(pl.Date, fmt='%Y-%m-%d'))
df_rates_ens = df_rates_ens.join(df_adj_factor, on='date', how='left')
df_rates_ens = df_rates_ens.with_columns((pl.col('rate') + pl.col('diff') * pl.col('adj_factor')).alias('rate'))
df_rates_ens = df_rates_ens.drop(['diff', 'adj_factor'])

# dampen the factors for further horizons
m = 0.90
df_rates_ens = df_rates_ens\
    .with_columns(pl.col('rate').shift().over(['cfips']).alias("lag_rate"))\
    .with_columns((pl.col('rate') - pl.col('lag_rate')).alias('diff_rate'))\
    .with_columns(pl.when(pl.col('diff_rate').is_null()).then(pl.col('rate')).otherwise(pl.col('diff_rate')).alias('diff_rate'))

df_damp_factor = pl.DataFrame({
    'date': ['2023-01-01', '2023-02-01', '2023-03-01', '2023-04-01', '2023-05-01', '2023-06-01'],
    'damp': [1.0, 1.0, m**1, m**2, m**3, m**4]}
).with_columns(pl.col('date').str.strptime(pl.Date, fmt='%Y-%m-%d'))
df_rates_ens = df_rates_ens.join(df_damp_factor, on='date')
df_rates_ens = df_rates_ens \
    .with_columns((pl.col('diff_rate') * pl.col('damp')).alias('diff_rate')) \
    .with_columns(pl.col('diff_rate').cumsum().over('cfips').alias('rate'))

# rates by horizon/date
list_stats = []
for d in df_rates_ens['date'].unique():
    stats = describe_numeric(df_rates_ens.filter(pl.col('date') == d).select(['rate']).to_pandas(), stats_nans=False)
    stats['date'] = d
    list_stats.append(stats)
stats = pd.concat(list_stats)
print(stats)

df_rates_ens = df_rates_ens.select(['cfips', 'asofdate', 'date', 'rate'])
dir_out = f'{config.DIR_ARTIFACTS}/combine_rates'
os.makedirs(dir_out, exist_ok=True)
file_out = f"{dir_out}/pub_ens_adj2pub_050_damp_090_add001.csv"
df_rates_ens.write_csv(file_out, float_precision=5)

print('done')
