import unittest
import plotly.offline as py

from tsfcst.forecasters.forecaster import Forecaster
from tsfcst.time_series import TsData
from tsfcst.utils import plot_fcsts_and_actual
from tsfcst.models.inventory import HoltWintersSmModel


class TestPlotManyForecasts(unittest.TestCase):

    def test_plot_many_forecasts(self):
        df_ts = TsData.sample_monthly()
        fcster = Forecaster(
            model_cls=HoltWintersSmModel,
            data=df_ts,
            boxcox_lambda=0.999,
            params_model={
                'damped_trend': True,
                'smoothing_level_max': 0.09,
                'smoothing_trend_max': 0.03,
                'smoothing_seasonal_max': 0.00001,
                'damping_trend_min': 0.95,
                'damping_trend_max': 0.98,
            })
        df_fcsts_cv, metrics_cv = fcster.cv(periods_test=3, include_last_date=True)
        fig = plot_fcsts_and_actual(df_ts.data, df_fcsts_cv)
        metrics_cv_str = Forecaster.metrics_cv_str_pretty(metrics_cv)
        fig.update_layout(title=f'Forecasts by HoltWintersSmModel', xaxis_title=metrics_cv_str)
        py.plot(fig)
        print(metrics_cv)
