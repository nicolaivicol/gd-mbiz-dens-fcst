import os
import pandas as pd
import numpy as np
import json
import logging
from tabulate import tabulate
import optuna
# import copy
# import plotly.offline as py

import config
# from tsfcst.time_series import TsData
from tsfcst.forecasters.forecaster import Forecaster
# from tsfcst.models.inventory import MovingAverageModel, HoltWintersSmModel, ProphetModel
from tsfcst.utils import smape_cv_opt  #, plot_fcsts_and_actual

log = logging.getLogger(os.path.basename(__file__))


class ParamsFinder:

    model_cls = None
    data = None

    @staticmethod
    def objective(trial):
        """ Objective function to tune forecaster model. """
        params_trial_forecaster = {**Forecaster.trial_params(trial)}
        params_trial_model = {**ParamsFinder.model_cls.trial_params(trial)}
        fcster = Forecaster(
            model_cls=ParamsFinder.model_cls,
            data=ParamsFinder.data,
            params_model=params_trial_model,
            **params_trial_forecaster,
        )
        df_fcsts_cv, metrics_cv = fcster.cv(periods_test=3)
        return smape_cv_opt(**metrics_cv)

    @staticmethod
    def tune_hyper_params_w_optuna(n_trials=120):
        log.info('START - Tune hyper-parameters')
        study = optuna.create_study(study_name=f'{ParamsFinder.model_cls.__name__}', direction='minimize')
        study.optimize(ParamsFinder.objective, n_trials=n_trials)

        metric_names = ['smape_cv_opt']
        df_cv_results = pd.DataFrame([dict(zip(metric_names, trial.values), **trial.params) for trial in study.trials])
        df_cv_results = df_cv_results.sort_values(by=['smape_cv_opt'], ascending=True).reset_index(drop=True)
        log.info(' - Best parameters found: \n' + json.dumps(study.best_params, indent=2))
        log.info(' - CV results: \n' + tabulate(df_cv_results.head(25), headers=df_cv_results.columns, showindex=False))

        df_cv_results.to_csv(config.FILE_TUNE_ALL_PARAMS_COMBS_CACHE, index=False, float_format='%.4f')
        with open(config.FILE_TUNE_PARAMS_BEST, 'w') as f:
            json.dump(study.best_params, f, indent=2)

        log.info('END - Tune hyper-parameters')
        return df_cv_results, study.best_params

    @staticmethod
    def best_params_top_median(df_cv_results, min_combs=1, max_combs=15, th_abs=0.50, th_prc=0.10):
        best_score = df_cv_results['smape_cv_opt'][0]
        th_score = best_score + max(th_abs, th_prc * best_score)
        is_close_score = np.array(df_cv_results['smape_cv_opt']) < th_score
        is_from_top = np.arange(1, len(df_cv_results) + 1) <= max_combs
        is_from_min_req = np.arange(1, len(df_cv_results) + 1) <= min_combs
        df_cv_results_top = df_cv_results.loc[is_from_min_req | (is_close_score & is_from_top)]

        types_ = dict(zip(df_cv_results_top.dtypes.index, df_cv_results_top.dtypes))
        best_params = {}

        for col, type_ in types_.items():
            if 'float' in str(type_):
                median_ = np.median(df_cv_results_top[col])
            elif 'int' in str(type_):
                median_ = int(np.median(df_cv_results_top[col]))
            elif 'bool' in str(type_):
                median_ = np.median(df_cv_results_top[col]) > 0.50
            elif str(type_) == 'object':
                median_ = df_cv_results_top[col].value_counts().nlargest(1).index[0]
            else:
                raise ValueError('unrecognized dtype')

            best_params[col] = median_

        best_metric = best_params.pop('smape_cv_opt')
        return best_metric, best_params

    @staticmethod
    def plot_parallel_optuna_res(df_res_tune=None):
        import plotly.offline as py
        import plotly.express as px

        if df_res_tune is None:
            df_res_tune = pd.read_csv(config.FILE_TUNE_ALL_PARAMS_COMBS_CACHE)

        fig = px.parallel_coordinates(
            df_res_tune,
            color='smape_cv_opt',
            dimensions=['smape_cv_opt'] + [col for col in df_res_tune.columns if col != 'smape_cv_opt'],
            color_continuous_scale=['green', 'yellow', 'red'],
            color_continuous_midpoint=np.median(df_res_tune['smape_cv_opt']),
        )
        # py.iplot(fig)
        py.plot(fig)

#
# if __name__ == "__main__":
#     model_cls = ProphetModel  # ProphetModel  # HoltWintersSmModel  # MovingAverageModel
#     data_ts = TsData.sample_monthly()
#
#     ParamsFinder.model_cls = model_cls
#     ParamsFinder.data = data_ts
#     df_cv_results, best_params = ParamsFinder.tune_hyper_params_w_optuna()
#
#     ParamsFinder.plot_parallel_optuna_res(df_cv_results.head(int(len(df_cv_results) * 0.33)))
#
#     best_params_median = df_cv_results.head(10).median()
#
#     df_ts = TsData.sample_monthly()
#     fcster = Forecaster(
#         model_cls=model_cls,
#         data=df_ts,
#         boxcox_lambda=best_params.pop('boxcox_lambda', None),
#         params_model=best_params
#     )
#     df_fcsts_cv, metrics_cv = fcster.cv(periods_test=3, include_last_date=True)
#     fig = plot_fcsts_and_actual(df_ts.data, df_fcsts_cv)
#     metrics_cv_str = Forecaster.metrics_cv_str_pretty(metrics_cv)
#     log.info(metrics_cv_str)
#     fig.update_layout(title=f'Forecasts by best model={model_cls.__name__}', xaxis_title=metrics_cv_str)
#     py.plot(fig)

