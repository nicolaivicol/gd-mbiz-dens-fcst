import os
import pandas as pd
import numpy as np
import json
import logging
from tabulate import tabulate
import optuna
from typing import List, Dict, Tuple
# import copy
# import plotly.offline as py

import config
from tsfcst.time_series import TsData
from tsfcst.models.abstract_model import TsModel
from tsfcst.forecasters.forecaster import Forecaster
from tsfcst.models.inventory import MovingAverageModel, HoltWintersSmModel, ProphetModel
from tsfcst.utils import smape_cv_opt  #, plot_fcsts_and_actual

log = logging.getLogger(os.path.basename(__file__))


class ParamsFinder:

    model_cls: type(TsModel) = None
    data: TsData = None

    @staticmethod
    def objective(trial: optuna.Trial):
        """
        Objective function to minimize.
        Parameters to choose from are the parameters of the forecasting model and the forecaster itself.
        Score to minimize is `smape_cv_opt` - SMAPE adjusted with penalties
        """
        params_trial_forecaster = ParamsFinder.get_trial_params(trial, Forecaster.trial_params())
        params_trial_model = ParamsFinder.get_trial_params(trial, ParamsFinder.model_cls.trial_params())
        fcster = Forecaster(
            model_cls=ParamsFinder.model_cls,
            data=ParamsFinder.data,
            params_model=params_trial_model,
            **params_trial_forecaster,
        )
        df_fcsts_cv, metrics_cv = fcster.cv(periods_test=3)
        return smape_cv_opt(**metrics_cv)

    @staticmethod
    def find_best(n_trials=120, id_cache='tmp', use_cache=False) -> Tuple[pd.DataFrame, Dict]:
        """ run many trials with various combinations of parameters to search for best parameters using optuna """

        file_df_trials = f'{config.DIR_CACHE_TUNE_HYPER_PARAMS_W_OPTUNA}/df_trials/{id_cache}.csv'
        file_best_params = f'{config.DIR_CACHE_TUNE_HYPER_PARAMS_W_OPTUNA}/best_params/{id_cache}.json'

        if use_cache and os.path.exists(file_df_trials) and os.path.exists(file_best_params):
            log.debug('find_best() - loading from cache')
            try:
                df_trials = pd.read_csv(file_df_trials)
                with open(file_best_params, 'r') as f:
                    best_result = json.load(f)
                return df_trials, best_result
            except Exception as e:
                log.warning('loading from cache failed: ' + str(e))

        log.info('find_best() - start search')
        study = optuna.create_study(study_name=f'{ParamsFinder.model_cls.__name__}', direction='minimize')
        study.optimize(ParamsFinder.objective, n_trials=n_trials)

        metric_names = ['smape_cv_opt']
        df_trials = pd.DataFrame([dict(zip(metric_names, trial.values), **trial.params) for trial in study.trials])
        df_trials = df_trials.sort_values(by=['smape_cv_opt'], ascending=True).reset_index(drop=True)
        best_result = {'best_value': study.best_value, 'best_params': study.best_params}

        # cache
        df_trials.to_csv(file_df_trials, index=False, float_format='%.4f')
        with open(file_best_params, 'w') as f:
            json.dump(best_result, f, indent=2)

        log.info(f'find_best() - best parameters found: \n'
                 + json.dumps(best_result, indent=2))
        log.info('find_best() - top 25 trials: \n' +
                 tabulate(df_trials.head(25), headers=df_trials.columns, showindex=False))

        return df_trials, best_result

    @staticmethod
    def best_params_top_median(df_cv_results, min_combs=1, max_combs=15, th_abs=0.50, th_prc=0.10):
        """ median/mode of top trials """

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
                raise ValueError(f'unrecognized dtype {str(type_)}')

            best_params[col] = median_

        best_metric = best_params.pop('smape_cv_opt')
        return best_metric, best_params

    @staticmethod
    def get_trial_params(trial: optuna.Trial, from_params_set: List[Dict]) -> Dict:
        params_trial = {}

        for par_definition in from_params_set:

            type_ = par_definition.pop('type')
            name = par_definition['name']

            if type_ == 'categorical':
                params_trial[name] = trial.suggest_categorical(**par_definition)
            elif type_ == 'int':
                params_trial[name] = trial.suggest_int(**par_definition)
            elif type_ == 'float':
                params_trial[name] = trial.suggest_float(**par_definition)
            else:
                raise ValueError(f"unrecognized type: '{type_}'")

        return params_trial

    @staticmethod
    def plot_parallel_optuna_res(df=None, use_express=False, plot=False):
        """ https://plotly.com/python/parallel-coordinates-plot/ """

        import plotly.offline as py
        import plotly.express as px
        import plotly.graph_objects as go

        if use_express:
            fig = px.parallel_coordinates(
                df,
                color='smape_cv_opt',
                dimensions=['smape_cv_opt'] + [col for col in df.columns if col != 'smape_cv_opt'],
                color_continuous_scale=['green', 'yellow', 'red'],
                color_continuous_midpoint=np.median(df['smape_cv_opt']),
                range_color=(0, np.max(df['smape_cv_opt']))
            )
            return fig

        names_ = ['smape_cv_opt'] + [col for col in df.columns if col != 'smape_cv_opt']
        dimensions_ = []
        pardefs = ParamsFinder.trial_params_definitions()

        for name_ in names_:
            pardef = pardefs.get(name_)
            if pardef is None:
                dim_ = dict(label=name_, values=df[name_])
                if name_ == 'smape_cv_opt':
                    dim_['range'] = [0, max(5, np.max(df[name_]))]
            elif pardef['type'] == 'categorical':
                map_ = {v: i for i, v in enumerate(pardef['choices'])}
                vals_numeric = [map_[v] for v in df[name_]]
                dim_ = dict(label=name_,
                            values=vals_numeric,
                            range=[-1, len(pardef['choices'])],
                            tickvals=[str(map_[v]) for v in pardef['choices']],
                            ticktext=pardef['choices']
                            )
            elif pardef['type'] in ['int', 'float']:
                dim_ = dict(label=name_, values=df[name_], range=[pardef['low'], pardef['high']])
                # if pardef.get('log', False):
                #     values_ = np.log1p(df[name_])
                #     range_ = [np.log1p(pardef['low']), np.log1p(pardef['high'])]
                #     tickvals = [np.min(values_) + i * (np.max(values_) - np.min(values_) / 5) for i in range(5)]
                #     tickvals.extend(range_)
                #     tickvals.sort()
                #     #np.arange(start=np.min(values_), stop=np.max(values_)*1.01, step=(np.max(values_) - np.min(values_))/5)
                #     ticktext = np.round(np.expm1(tickvals), 3)
                #     # ticks_ = ticks_[ticks_ < np.max(df[name_])]
                #     dim_ = dict(label=name_, values=values_, range=range_, tickvals=tickvals, ticktext=ticktext,
                #                 visible=True)
                # else:
                #     dim_ = dict(label=name_, values=df[name_], range=[pardef['low'], pardef['high']])

            else:
                raise ValueError('unrecognized type:' + pardef['type'])

            dimensions_.append(dim_)

        fig = go.Figure(
            data=go.Parcoords(
                line=dict(
                    color=df['smape_cv_opt'],
                    colorscale=[[0.0, 'green'], [0.25, 'yellow'], [0.50, 'red'], [1, 'darkred']],
                    showscale=True, cmin=0, cmax=25,
                ),
                dimensions=dimensions_
            )
        )

        if plot:
            # py.iplot(fig)
            py.plot(fig)

        return fig


    @staticmethod
    def trial_params_definitions() -> Dict:

        pardefs = MovingAverageModel.trial_params() \
                  + HoltWintersSmModel.trial_params() \
                  + ProphetModel.trial_params() \
                  + Forecaster.trial_params()

        pardefs_dict = {}

        for pardef in pardefs:
            name_ = pardef['name']
            pardefs_dict[name_] = pardef

        return pardefs_dict
