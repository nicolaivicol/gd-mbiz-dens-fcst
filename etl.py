import polars as pl

import config


def load_raw_data():
    dtypes_train = {
        'row_id': str,
        'cfips': str,
        'county': str,
        'state': str,
        'first_day_of_month': pl.Date,
        'microbusiness_density': pl.Float64,
        'active': pl.Int64
    }
    df_train = pl.read_csv(f'{config.DIR_DATA}/train.csv', dtypes=dtypes_train)
    df_train.describe().to_pandas()

    dtypes_test = {
        'row_id': str,
        'cfips': str,
        'first_day_of_month': pl.Date
    }
    df_test = pl.read_csv(f'{config.DIR_DATA}/test.csv', dtypes=dtypes_test)

    dtypes_census = {
        'pct_bb_2017': pl.Float64,
        'pct_bb_2018': pl.Float64,
        'pct_bb_2019': pl.Float64,
        'pct_bb_2020': pl.Float64,
        'pct_bb_2021': pl.Float64,
        'cfips': str,
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

    df_train = df_train \
        .with_column((pl.col('active') * 100 / pl.col('microbusiness_density')).round(0).alias('population'))

    return df_train, df_test, df_census






