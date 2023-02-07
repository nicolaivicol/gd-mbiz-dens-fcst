import unittest
import pandas as pd

from tsfcst.forecasters.forecaster import Forecaster
from tsfcst.time_series import TsData


class TestForecaster(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.df_ts = TsData.sample_monthly()
        self.fcster = Forecaster.from_named_configs(data=self.df_ts, config_name='hw')

    def test_forecast(self):
        periods_ahead = 6
        fcst = self.fcster.forecast(periods_ahead=periods_ahead, train_date='2022-06-01')
        isinstance(fcst, pd.DataFrame)
        assert fcst.shape[0] == periods_ahead
        print(fcst.head(periods_ahead))

    def test_cv(self):
        df_fcsts_cv, metrics_cv = self.fcster.cv(
            n_train_dates=3,
            step_train_dates=2,
            periods_val=5,
            periods_test=3,
            periods_out=7
        )
        isinstance(df_fcsts_cv, pd.DataFrame)
        assert df_fcsts_cv.shape[0] > 0
        assert isinstance(metrics_cv, dict)
        print(metrics_cv)
