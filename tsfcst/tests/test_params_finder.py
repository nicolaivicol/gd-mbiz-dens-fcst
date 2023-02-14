import unittest
import plotly.offline as py

from tsfcst.time_series import TsData
from tsfcst.models.inventory import ThetaSmModel
from tsfcst.params_finder import ParamsFinder
from tsfcst.forecasters.forecaster import Forecaster
from tsfcst.utils_tsfcst import plot_fcsts_and_actual


class TestParamsFinder(unittest.TestCase):

    def test_ParamsFinder(self):
        model_cls = ThetaSmModel
        ts = TsData.sample_monthly()

        ParamsFinder.model_cls = model_cls
        ParamsFinder.data = ts
        df_trials, best_result, param_importances = ParamsFinder.find_best(n_trials=50, use_cache=False)
        print('best_params: \n' + str(best_result))

        best_metric, best_params_median = ParamsFinder.best_params_top_median(df_trials)
        print('best_params_median: \n' + str(best_params_median))

        print('top trials: \n' + str(df_trials.head(5)))
        py.plot(ParamsFinder.plot_parallel_optuna_res(df_trials.head(max(int(len(df_trials) * 0.33), 25))),
                filename='temp-plot_parallel_optuna_res.html')

        print('importances: \n' + str(param_importances))
        py.plot(ParamsFinder.plot_importances(param_importances),
                filename='temp-plot_importances.html')

        fcster = Forecaster(
            model_cls=model_cls,
            data=ts,
            boxcox_lambda=best_params_median.pop('boxcox_lambda', None),
            use_data_since=best_params_median.pop('use_data_since', None),
            params_model=best_params_median
        )
        df_fcsts_cv, metrics_cv = fcster.cv(periods_test=3, periods_out=7)

        fig = plot_fcsts_and_actual(ts.data, df_fcsts_cv)
        metrics_cv_str = Forecaster.metrics_cv_str_pretty(metrics_cv)
        print(metrics_cv_str)
        fig.update_layout(title=f'Forecasts by best model={model_cls.__name__}', xaxis_title=metrics_cv_str)
        py.plot(fig, filename='temp-plot_fcsts_and_actual.html')
