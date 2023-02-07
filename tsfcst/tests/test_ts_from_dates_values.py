import unittest
import pandas as pd
import polars as pl

from tsfcst.time_series import from_dates_values
from tsfcst.tests import utils as test_utils


class TestTsFromDatesValues(unittest.TestCase):

    def test_load_from_list_of_values_monthly(self):
        ts = from_dates_values(dates=['2021-01-01', '2021-02-01'], values=[1, 2], freq='M')
        test_utils.is_time_series(ts)
        self.assertEqual([1, 2], list(ts['value']))
        assert list(ts.index) == list(pd.to_datetime(['2021-01-01', '2021-02-01']))

    def test_load_from_list_of_values_monthly_bad_order(self):
        ts = from_dates_values(dates=['2021-02-01', '2021-01-01'], values=[2, 1], freq='M')
        test_utils.is_time_series(ts)
        self.assertEqual([1, 2], list(ts['value']))
        assert list(ts.index) == list(pd.to_datetime(['2021-01-01', '2021-02-01']))

    def test_load_as_dataframe(self):
        ts = from_dates_values(dates=['2021-02-01', '2021-01-01'], values=[2, 1], freq='M', as_df=True)
        test_utils.is_time_series(ts)
        self.assertEqual([1, 2], list(ts['value']))
        assert list(ts['date']) == list(pd.to_datetime(['2021-01-01', '2021-02-01']))

    def test_load_from_polars(self):
        df = pl.DataFrame({'date': ['2021-02-01', '2021-01-01'], 'value': [2, 1]})
        df = df.with_column(pl.col('date').str.strptime(pl.Date))
        ts = from_dates_values(dates=df['date'], values=df['value'], freq='MS', as_df=True)
        test_utils.is_time_series(ts)
        self.assertEqual([1, 2], list(ts['value']))
        assert list(ts['date']) == list(pd.to_datetime(['2021-01-01', '2021-02-01']))
