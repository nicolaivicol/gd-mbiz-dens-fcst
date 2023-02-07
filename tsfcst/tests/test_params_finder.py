import unittest
import plotly.offline as py

from tsfcst.time_series import TsData
from tsfcst.models.inventory import MovingAverageModel, HoltWintersSmModel, ProphetModel
from tsfcst.params_finder import ParamsFinder
from tsfcst.forecasters.forecaster import Forecaster
from tsfcst.utils import plot_fcsts_and_actual


class TestParamsFinder(unittest.TestCase):

    def test_ParamsFinder(self):
        model_cls = HoltWintersSmModel
        df_ts = TsData.sample_monthly()

        ParamsFinder.model_cls = model_cls
        ParamsFinder.data = df_ts
        df_trials, best_params = ParamsFinder.find_best(n_trials=100, use_cache=True)
        print('best_params: \n' + str(best_params))

        best_metric, best_params_median = ParamsFinder.best_params_top_median(df_trials)
        print('best_params_median: \n' + str(best_params_median))

        ParamsFinder.plot_parallel_optuna_res(df_trials.head(int(len(df_trials) * 0.33)))

        fcster = Forecaster(
            model_cls=model_cls,
            data=df_ts,
            boxcox_lambda=best_params_median.pop('boxcox_lambda', None),
            params_model=best_params_median
        )
        df_fcsts_cv, metrics_cv = fcster.cv(periods_test=3, periods_out=7)

        fig = plot_fcsts_and_actual(df_ts.data, df_fcsts_cv)
        metrics_cv_str = Forecaster.metrics_cv_str_pretty(metrics_cv)
        print(metrics_cv_str)
        fig.update_layout(title=f'Forecasts by best model={model_cls.__name__}', xaxis_title=metrics_cv_str)
        py.plot(fig)
