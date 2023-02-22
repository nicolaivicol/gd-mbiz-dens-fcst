import os
import logging
import random
import plotly.graph_objects as go
import plotly.offline as py
import numpy as np
import polars as pl
from datetime import date, datetime

import config
from etl import load_data
from utils import describe_numeric, set_display_options

set_display_options()
log = logging.getLogger(os.path.basename(__file__))

WEIGHTS_LAST_3Y = [0.00, 0.00, 0.00]
# WEIGHTS_LAST_3Y = [0.15, 0.25, 0.30]
# WEIGHTS_LAST_3Y = [0.20, 0.30, 0.35]
# WEIGHTS_LAST_3Y = [0.25, 0.35, 0.40]

df_train, _, _ = load_data()

# manual fix when active is zero and population can't be inferred
map_replace_nans = [
    {'cfips': 28055, 'from_date': '2021-02-01', 'to_date': '2021-12-01', 'value': 1162},
    {'cfips': 28055, 'from_date': '2022-01-01', 'to_date': '2022-10-01', 'value': 1185},
    {'cfips': 48301, 'from_date': '2020-01-01', 'to_date': '2020-03-01', 'value': 78},
    {'cfips': 48301, 'from_date': '2021-02-01', 'to_date': '2022-03-01', 'value': 73},
]

for d in map_replace_nans:
    cfips, from_date, to_date, value = d.values()
    df_train = df_train.with_columns(
        pl.when((pl.col('cfips') == cfips)
                & pl.col('first_day_of_month').is_between(datetime.strptime(from_date, "%Y-%m-%d").date(),
                                                          datetime.strptime(to_date, "%Y-%m-%d").date())
                & pl.col('population').is_nan())
        .then(pl.lit(value))
        .otherwise(pl.col('population'))
        .alias('population'))

df_pop = df_train \
    .select(['cfips', 'first_day_of_month', 'population']) \
    .with_columns([pl.col('population').shift(1).over('cfips').alias('lag_population')]) \
    .with_columns([pl.col('first_day_of_month').dt.strftime('%m').cast(pl.Int8).alias('month')]) \
    .filter(pl.col('month') == 1) \
    .drop('month') \
    .with_columns((pl.col('population') / pl.col('lag_population') - 1).alias('prc_change'))

df_weights = pl.DataFrame(
    {'first_day_of_month': ['2020-01-01', '2021-01-01', '2022-01-01'],
     'weight': WEIGHTS_LAST_3Y})\
    .with_columns(pl.col('first_day_of_month').str.strptime(pl.Date, fmt='%Y-%m-%d'))

df_pop_2023 = df_pop \
    .join(df_weights, on='first_day_of_month', how='left') \
    .groupby('cfips') \
    .agg([pl.col('population').last().alias('lag_population'),
          (pl.col('prc_change') * pl.col('weight')).sum().alias('prc_change')]) \
    .with_columns([(pl.col('lag_population') * (1 + pl.col('prc_change'))).round(2).alias('population'),
                   pl.lit('2023-01-01').str.strptime(pl.Date, fmt='%Y-%m-%d').alias('first_day_of_month')])

df_pop = pl.concat([df_pop, df_pop_2023.select(df_pop.columns)])\
    .with_columns(pl.col('first_day_of_month').dt.strftime('%Y').cast(pl.Int16).alias('year'))\
    .sort(['cfips', 'first_day_of_month'])

df_dates = pl.DataFrame({'misc': 1, 'first_day_of_month': pl.date_range(date(2019, 8, 1), date(2023, 6, 1), "1mo")})\
    .with_columns(pl.col('first_day_of_month').dt.strftime('%Y').cast(pl.Int16).alias('year'))
df_cfips = pl.DataFrame({'misc': 1, 'cfips': np.unique(df_train['cfips'])})

df_full = df_cfips \
    .join(df_dates, on='misc') \
    .join(df_train.select(['cfips', 'first_day_of_month', 'population']),
          on=['cfips', 'first_day_of_month'], how='left') \
    .join(df_pop, on=['cfips', 'year'], how='left', suffix='_est') \
    .with_columns(pl.when(pl.col('population').is_null())
                  .then(pl.col('population_est'))
                  .otherwise(pl.col('population'))
                  .alias('population')) \
    .select(['cfips', 'first_day_of_month', 'population'])

print(describe_numeric(df_full.to_pandas()))

id = '_'.join([str(int(w*100)) for w in WEIGHTS_LAST_3Y])
df_full.write_csv(f'{config.DIR_DATA}/est_pop_{id}.csv')
log.debug('saved estimated population to: ' + f'{config.DIR_DATA}/est_pop_{id}.csv')

fig = go.Figure()
for _ in range(100):
    cfips = random.choice(np.unique(df_train['cfips']))
    tmp = df_full.filter(pl.col('cfips') == cfips)
    fig.add_trace(
        go.Scatter(
            x=tmp['first_day_of_month'],
            y=tmp['population'],
            name=str(cfips),
            mode='lines+markers',
            opacity=0.7
        )
    )
fig.update_layout(title='populations')
py.plot(fig, filename=f'temp-plot-est_pop_{id}.html')
