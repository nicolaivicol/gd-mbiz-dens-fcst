import itertools
import polars as pl
import pandas as pd
from datetime import date, datetime

import config


def load_raw_data():
    dtypes_train = {
        'row_id': str,
        'cfips': pl.Int64,
        'county': str,
        'state': str,
        'first_day_of_month': pl.Date,
        'microbusiness_density': pl.Float64,
        'active': pl.Int64
    }
    df_train = pl.read_csv(f'{config.DIR_DATA}/train.csv', dtypes=dtypes_train)
    df_revealed_test = pl.read_csv(f'{config.DIR_DATA}/revealed_test.csv', dtypes=dtypes_train)
    df_train = pl.concat([df_train, df_revealed_test])

    dtypes_test = {
        'row_id': str,
        'cfips': pl.Int64,
        'first_day_of_month': pl.Date
    }
    df_test = pl.read_csv(f'{config.DIR_DATA}/test.csv', dtypes=dtypes_test)

    dtypes_census = {
        'pct_bb_2017': pl.Float64,
        'pct_bb_2018': pl.Float64,
        'pct_bb_2019': pl.Float64,
        'pct_bb_2020': pl.Float64,
        'pct_bb_2021': pl.Float64,
        'cfips': pl.Int64,
        'pct_college_2017': pl.Float64,
        'pct_college_2018': pl.Float64,
        'pct_college_2019': pl.Float64,
        'pct_college_2020': pl.Float64,
        'pct_college_2021': pl.Float64,
        'pct_foreign_born_2017': pl.Float64,
        'pct_foreign_born_2018': pl.Float64,
        'pct_foreign_born_2019': pl.Float64,
        'pct_foreign_born_2020': pl.Float64,
        'pct_foreign_born_2021': pl.Float64,
        'pct_it_workers_2017': pl.Float64,
        'pct_it_workers_2018': pl.Float64,
        'pct_it_workers_2019': pl.Float64,
        'pct_it_workers_2020': pl.Float64,
        'pct_it_workers_2021': pl.Float64,
        'median_hh_inc_2017': pl.Float64,
        'median_hh_inc_2018': pl.Float64,
        'median_hh_inc_2019': pl.Float64,
        'median_hh_inc_2020': pl.Float64,
        'median_hh_inc_2021': pl.Float64
    }
    df_census = pl.read_csv(f'{config.DIR_DATA}/census_starter.csv', dtypes=dtypes_census)

    return df_train, df_test, df_census


def load_data():
    df_train, df_test, df_census = load_raw_data()
    df_train = df_train.sort(['cfips', 'first_day_of_month'])
    df_train = df_train.with_columns((pl.col('active') * 100 / pl.col('microbusiness_density')).round(0).alias('population'))
    df_pop = load_population(df_train)
    return df_train, df_test, df_census, df_pop


def get_df_ts_by_cfips(cfips, target_name, _df: pl.DataFrame):
    return _df.filter(pl.col('cfips') == int(cfips)).select(['first_day_of_month', target_name])


def get_df_ts_by_state(state, target_name, _df: pl.DataFrame):
    if state not in ['', 'all']:
        df_filt = _df.filter(pl.col('state') == state)
    else:
        df_filt = _df.filter(pl.col('state') != '')

    if target_name == 'active':
        df_ts = df_filt.groupby('first_day_of_month').agg(pl.sum(target_name)).sort('first_day_of_month')
    else:
        df_ts = df_filt.groupby('first_day_of_month').agg(pl.mean(target_name)).sort('first_day_of_month')

    return df_ts


def get_df_ts_by_id(idcol, id, target_name, _df: pl.DataFrame):
    if idcol == 'cfips':
        return get_df_ts_by_cfips(id, target_name, _df)
    elif idcol == 'state':
        return get_df_ts_by_state(id, target_name, _df)
    else:
        raise ValueError(f'idcol={idcol} not recognized')


def load_df_ts_by_cfips(cfips, target_name):
    df, _, _, _ = load_data()
    df_ts = get_df_ts_by_cfips(cfips=cfips, target_name=target_name, _df=df)
    return df_ts


def population_2021_for_2023():
    df = pd.read_csv(f'{config.DIR_DATA}/ACSST5Y2021.S0101-Data.csv',
                     usecols=['GEO_ID', 'NAME', 'S0101_C01_026E'],
                     low_memory=False)
    df = df.iloc[1:]
    df['population'] = df['S0101_C01_026E'].astype('int')
    df['cfips'] = df.GEO_ID.apply(lambda x: int(x.split('US')[-1]))
    df = pl.from_pandas(df[['cfips', 'population']]) \
        .with_columns([
            pl.col('population').cast(pl.Float64),
            pl.lit('2023-01-01').str.strptime(pl.Date, fmt='%Y-%m-%d').alias('first_day_of_month')
        ]) \
        .select(['cfips', 'first_day_of_month', 'population'])
    return df


def load_population(df_train):
    df_pop_train = df_train \
        .with_columns(pl.col('first_day_of_month').dt.strftime('%Y').alias('year')) \
        .with_columns(pl.col('population').fill_nan(None).mean().over(['cfips', 'year'])) \
        .select(['cfips', 'first_day_of_month', 'population'])

    df_pop_for_2023 = population_2021_for_2023() \
        .rename({'first_day_of_month': 'start_of_year'}) \
        .join(pl.DataFrame({'start_of_year': date(2023, 1, 1),
                            'first_day_of_month': pl.date_range(date(2023, 1, 1), date(2023, 6, 1), "1mo")}),
              on='start_of_year') \
        .join(df_pop_train.select(['cfips']).unique(), on='cfips') \
        .select(['cfips', 'first_day_of_month', 'population'])

    df_pop = pl.concat([df_pop_train, df_pop_for_2023]).sort(['cfips', 'first_day_of_month'])

    return df_pop


if __name__ == '__main__':
    print('')

