import unittest

from tsfcst.config_selector import ConfigSelector
from tsfcst.forecasters.forecaster import Forecaster
from tsfcst.time_series import TsData


class TestConfigSelector(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.df_ts = TsData.sample_monthly()
        self.fcster = Forecaster.from_named_configs(data=self.df_ts, config_name='hw')

    def test_selector(self):
        ts = TsData(self.df_ts.time, self.df_ts.target)
        cs = ConfigSelector(ts)
        slct = cs.select(n_train_dates=3, step_train_dates=3, periods_ahead=6, metric='smape_avg_val')
        print(slct)
        print(cs.cv_res)

    # def test_selector_with_model(self):
    #     ts = TsData(self.df_ts['date'], self.df_ts['sales_usd'])
    #     cs = ConfigSelector(ts, model_error_weight=0.99)
    #     slct = cs.select(train_dates_n=3, train_dates_freq=60, periods_ahead=180)
    #     print(slct)
    #     print(cs.cv_res)
