import unittest
import pandas as pd

from tsfcst.models.inventory import ThetaSmModel
from tsfcst.forecasters.forecaster import Forecaster, ForecasterConfig
from tsfcst.time_series import TsData


class TestForecaster(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        df_ts = TsData.sample_monthly()
        cfg = ForecasterConfig(ThetaSmModel, {}, {})
        self.df_ts = df_ts
        self.fcster = Forecaster.from_config(data=df_ts, cfg=cfg)

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
        assert isinstance(df_fcsts_cv, pd.DataFrame)
        assert df_fcsts_cv.shape[0] > 0
        assert isinstance(metrics_cv, dict)
        print(metrics_cv)

    def test_cv_without_test_out(self):
        df_fcsts_cv, metrics_cv = self.fcster.cv(
            n_train_dates=3,
            step_train_dates=2,
            periods_val=7,
            periods_test=0,
            periods_out=0,
            periods_val_last=5,
            offset_last_date=3
        )
        assert isinstance(df_fcsts_cv, pd.DataFrame)
        assert df_fcsts_cv.shape[0] > 0
        assert isinstance(metrics_cv, dict)
        print(metrics_cv)
