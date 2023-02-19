import os
import logging
import random
import plotly.graph_objects as go
import plotly.offline as py
import numpy as np
import polars as pl
from datetime import date

import config
from etl import load_data


log = logging.getLogger(os.path.basename(__file__))

WEIGHTS_LAST_3Y = [0.20, 0.30, 0.35]
#WEIGHTS_LAST_3Y = [0.00, 0.00, 0.00]

df_train, _, _ = load_data()

# manual fix when active is zero and population can't be infered
df_train = df_train.with_columns(
    pl.when((pl.col('cfips') == 28055) & pl.col('population').is_nan())
    .then(pl.lit(1185))
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


id = '_'.join([str(int(w*100)) for w in WEIGHTS_LAST_3Y])
df_full.write_csv(f'{config.DIR_DATA}/est_pop_{id}.csv')

cfips = random.choice(np.unique(df_train['cfips']))
tmp = df_full.filter(pl.col('cfips') == cfips)

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=tmp['first_day_of_month'],
        y=tmp['population'],
        name=f'population',
        mode='lines+markers',
        opacity=0.7,
        line=dict(color='black', width=2))
)
fig.update_layout(title=f'cfips={cfips}')
py.plot(fig)
