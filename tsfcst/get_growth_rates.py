import os
import polars as pl

import config
from utils import set_display_options
from etl import load_data

set_display_options()

map_id_run_asofdate = [
    ['2022-07-01', 'active-state-20220701-theta-test-trend_level_damp-100-0_02'],
    ['2022-12-01', 'active-state-20221201-theta-full-trend_level_damp-100-0_02'],
]

list_rates = []
for asofdate, id_run_model in map_id_run_asofdate:
    # asofdate, id_run_model = map_id_run_asofdate[0]
    file_df_fcsts_cv = f'{config.DIR_ARTIFACTS}/test_best_params/{id_run_model}/df_fcsts_cv.csv'
    df_model = pl.read_csv(file_df_fcsts_cv, parse_dates=True)

    df_actual_last = df_model \
        .with_columns(pl.lit(asofdate).str.strptime(pl.Date, fmt='%Y-%m-%d').alias('asofdate')) \
        .filter(pl.col('date') == pl.lit(asofdate).str.strptime(pl.Date, fmt='%Y-%m-%d')) \
        .select(['state', 'asofdate', 'actual'])

    col_fcst = f"fcst_{asofdate.replace('-', '')}"
    df_fcst = df_model \
        .filter((pl.col('date') > pl.lit(asofdate).str.strptime(pl.Date, fmt='%Y-%m-%d'))
                & pl.col(col_fcst).is_not_null()) \
        .select(['state', 'date', col_fcst])

    df_rates = df_fcst.join(df_actual_last, on='state')
    df_rates = df_rates \
        .with_columns((pl.col(col_fcst)/pl.col('actual') - 1).alias('rate'))\
        .select(['state', 'asofdate', 'date', 'rate'])

    df_rates = df_rates.filter(pl.col('state') != 'all') \
        .join(df_rates.filter(pl.col('state') == 'all')
              .select(['date', 'rate'])
              .rename({'rate': 'rate_country'}),
              on='date')\
        .rename({'rate': 'rate_state'})

    list_rates.append(df_rates)

df_rates = pl.concat(list_rates)
print(df_rates.filter(pl.col('state') == 'Alabama').to_pandas())

df_train, df_test, df_census, df_pop = load_data()
df_cfips_state = df_train.select(['cfips', 'state']).unique()

df_rates = df_cfips_state.join(df_rates, on='state').sort(['cfips', 'asofdate', 'date'])
print(df_rates.filter(pl.col('cfips') == 1001).to_pandas())

dir_out = f'{config.DIR_ARTIFACTS}/get_growth_rates'
os.makedirs(dir_out, exist_ok=True)
file_out = f'{dir_out}/rates-theta.csv'
df_rates.write_csv(file_out, float_precision=4)

print(f'rates saved to: {file_out}')
