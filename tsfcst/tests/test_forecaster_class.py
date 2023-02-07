import unittest
import pandas as pd

from tsfcst.config_selector import ConfigSelector
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
            train_dates_n=3,
            train_dates_freq=2,
            periods_ahead=5,
            periods_test=3,
            include_last_date=True
        )
        isinstance(df_fcsts_cv, pd.DataFrame)
        assert df_fcsts_cv.shape[0] > 0
        assert isinstance(metrics_cv, dict)
        print(metrics_cv)

    def test_selector(self):
        ts = TsData(self.df_ts.time, self.df_ts.target)
        cs = ConfigSelector(ts)
        slct = cs.select(train_dates_n=3, train_dates_freq=3, periods_ahead=6, metric='smape_avg_out')
        print(slct)
        print(cs.cv_res)

    # def test_selector_with_model(self):
    #     ts = TsData(self.df_ts['date'], self.df_ts['sales_usd'])
    #     cs = ConfigSelector(ts, model_error_weight=0.99)
    #     slct = cs.select(train_dates_n=3, train_dates_freq=60, periods_ahead=180)
    #     print(slct)
    #     print(cs.cv_res)
