import os
import pandas as pd
import numpy as np
import json
import logging
from tabulate import tabulate
import optuna
from typing import List, Dict, Tuple
import random

import config
from tsfcst.time_series import TsData
from tsfcst.models.abstract_model import TsModel
from tsfcst.forecasters.forecaster import Forecaster
from tsfcst.models.inventory import MODELS
from tsfcst.utils_tsfcst import smape_cv_opt

log = logging.getLogger(os.path.basename(__file__))
optuna.logging.disable_propagation()
optuna.logging.disable_default_handler()


class ParamsFinder:
    """
    Find best hyper-parameters of the model based on CV. Search performed with optuna.
    https://towardsdatascience.com/how-to-make-your-model-awesome-with-optuna-b56d490368af
    """

    model_cls: type(TsModel) = None
    data: TsData = None
    name_objective = 'smape_cv_opt'

    # for search space
    trend = True
    seasonal = True
    multiplicative = True
    level = True
    damp = False

    # for cv
    reg_coef = 0.0
    n_train_dates = None
    step_train_dates = None
    periods_val = None
    periods_test = 0
    periods_out = 0
    periods_val_last = None

    # ovverides
    choices_use_data_since = None

    # cache during find_best() to avoid repeated calculations
    _cache = {}

    @staticmethod
    def objective(trial: optuna.Trial):
        """
        Objective function to minimize.
        Parameters to choose from are the parameters of the forecasting model and the forecaster itself.
        Score to minimize is `smape_cv_opt` - SMAPE adjusted with penalties
        """
        forecaster_trial_params = Forecaster.trial_params(ParamsFinder.choices_use_data_since)
        params_trial_forecaster = ParamsFinder.get_trial_params(trial, forecaster_trial_params)
        params_trial_model_narrowed = ParamsFinder.model_cls.trial_params(
            trend=ParamsFinder.trend,
            seasonal=ParamsFinder.seasonal,
            multiplicative=ParamsFinder.multiplicative,
            level=ParamsFinder.level,
            damp=ParamsFinder.damp,
        )
        params_trial_model = ParamsFinder.get_trial_params(trial, params_trial_model_narrowed)
        hash_cache = json.dumps({**params_trial_forecaster, **params_trial_model}, sort_keys=True)
        out = ParamsFinder._cache.get(hash_cache, None)
        if out is not None:
            return out
        out = ParamsFinder.smape_cv_opt_penalized(params_trial_forecaster, params_trial_model)
        ParamsFinder._cache[hash_cache] = out
        return out

    @staticmethod
    def smape_cv_opt_penalized(params_forecaster: Dict, params_model: Dict):
        fcster = Forecaster(
            model_cls=ParamsFinder.model_cls,
            data=ParamsFinder.data,
            params_model=params_model,
            **params_forecaster,
        )
        try:
            df_fcsts_cv, metrics_cv = fcster.cv(
                n_train_dates=ParamsFinder.n_train_dates,
                step_train_dates=ParamsFinder.step_train_dates,
                periods_val=ParamsFinder.periods_val,
                periods_test=ParamsFinder.periods_test,
                periods_out=ParamsFinder.periods_out,
                periods_val_last=ParamsFinder.periods_val_last
            )
            smape_cv_opt_ = smape_cv_opt(**metrics_cv)
            model_flexibility_ = fcster.model.flexibility()
        except Exception as e:
            log.error(f'Error: {str(e)}')
            smape_cv_opt_ = 99999
            model_flexibility_ = 0

        return smape_cv_opt_ + ParamsFinder.reg_coef * model_flexibility_

    @staticmethod
    def trial_params_grid(n_trials_grid: int = None):
        model_params_grid = ParamsFinder.model_cls.trial_params_grid(ParamsFinder.model_cls.trial_params())
        forecaster_params_grid = Forecaster.trial_params_grid(Forecaster.trial_params())

        if n_trials_grid is None:
            n_trials_grid = len(forecaster_params_grid) * len(model_params_grid)

        n_random_trials_max = len(forecaster_params_grid) * len(model_params_grid)
        n_trials_grid = min(n_trials_grid, n_random_trials_max)
        idx_to_run = sorted(random.sample(range(n_random_trials_max), n_trials_grid))

        trials = []
        i = -1
        for forecaster_params in forecaster_params_grid:
            for model_params in model_params_grid:
                i += 1
                if i not in idx_to_run:
                    continue
                trial = optuna.trial.create_trial(
                    params={**forecaster_params['params'], **model_params['params']},
                    distributions={**forecaster_params['distributions'], **model_params['distributions']},
                    value=ParamsFinder.smape_cv_opt_penalized(forecaster_params['params'], model_params['params']),
                )
                trials.append(trial)

        return trials

    @staticmethod
    def get_file_names_cache(id_cache):
        file_df_trials = f'{config.DIR_CACHE_TUNE_HYPER_PARAMS_W_OPTUNA}/df_trials/{id_cache}.csv'
        file_best_params = f'{config.DIR_CACHE_TUNE_HYPER_PARAMS_W_OPTUNA}/best_params/{id_cache}.json'
        file_param_importances = f'{config.DIR_CACHE_TUNE_HYPER_PARAMS_W_OPTUNA}/param_importances/{id_cache}.csv'
        return file_df_trials, file_best_params, file_param_importances

    @staticmethod
    def cache_exists(id_cache):
        file_df_trials, file_best_params, file_param_importances = ParamsFinder.get_file_names_cache(id_cache)
        return os.path.exists(file_df_trials) and os.path.exists(file_best_params) and os.path.exists(file_param_importances)

    @staticmethod
    def find_best(n_trials=100, id_cache='tmp', use_cache=False, parimp=True, n_trials_grid=0
                  ) -> Tuple[pd.DataFrame, Dict, pd.DataFrame]:
        """ run many trials with various combinations of parameters to search for best parameters using optuna """

        assert (n_trials + n_trials_grid) > 0

        ParamsFinder._cache = {}

        file_df_trials, file_best_params, file_param_importances = ParamsFinder.get_file_names_cache(id_cache)

        if use_cache and ParamsFinder.cache_exists(id_cache):
            log.debug('find_best() - loading from cache')
            try:
                df_trials = pd.read_csv(file_df_trials)
                param_importances = pd.read_csv(file_param_importances)
                with open(file_best_params, 'r') as f:
                    best_result = json.load(f)
                return df_trials, best_result, param_importances
            except Exception as e:
                log.warning('loading from cache failed: ' + str(e))

        log.info('find_best() - start search')
        model_cls_name = ParamsFinder.model_cls.__name__

        study = optuna.create_study(study_name=f'{model_cls_name}', direction='minimize')
        if n_trials_grid > 0:
            study.add_trials(ParamsFinder.trial_params_grid(n_trials_grid))
        if n_trials > 0:
            study.optimize(ParamsFinder.objective, n_trials=n_trials)

        metric_names = ['smape_cv_opt']
        df_trials = pd.DataFrame([dict(zip(metric_names, trial.values), **trial.params) for trial in study.trials])
        df_trials = df_trials.sort_values(by=['smape_cv_opt'], ascending=True).reset_index(drop=True)
        best_result = {'best_value': study.best_value, 'best_params': study.best_params}

        param_importances = None
        if parimp:
            try:
                param_importances = optuna.importance.get_param_importances(study)
                param_importances = pd.DataFrame({'parameter': param_importances.keys(),
                                                  'importance': param_importances.values()})
            except RuntimeError as e:
                pass

        # cache
        if use_cache:
            df_trials.to_csv(file_df_trials, index=False, float_format='%.4f')
            with open(file_best_params, 'w') as f:
                json.dump(best_result, f, indent=2)
            if param_importances is not None:
                param_importances.to_csv(file_param_importances, index=False, float_format='%.4f')

        log.info(f'find_best() - best parameters found: \n'
                 + json.dumps(best_result, indent=2))
        log.info('find_best() - top 10 trials: \n' +
                 tabulate(df_trials.head(10), headers=df_trials.columns, showindex=False))

        return df_trials, best_result, param_importances

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
        pardefs = ParamsFinder.trial_params_definitions(ParamsFinder.model_cls)

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
    def plot_importances(df: pd.DataFrame, plot=False):

        import plotly.offline as py
        import plotly.graph_objects as go
        import plotly.colors

        layout = go.Layout(
            title="Hyperparameter Importances",
            xaxis={"title": "Importance"},
            yaxis={"title": "Hyperparameter"},
            showlegend=False,
        )

        if df is None or len(df) == 0:
            return go.Figure(data=[], layout=layout)

        df = df.sort_values(['importance'], ascending=True)

        fig = go.Figure(
            data=[
                go.Bar(
                    x=df['importance'],
                    y=df['parameter'],
                    marker_color=plotly.colors.sequential.Blues[-4],
                    orientation='h',
                )
            ],
            layout=layout,
        )

        if plot:
            # py.iplot(fig)
            py.plot(fig)

        return fig


    @staticmethod
    def trial_params_definitions(model_cls=None) -> Dict:

        pardefs = Forecaster.trial_params()
        if model_cls is None:
            for model_cls in MODELS.values():
                pardefs.extend(model_cls.trial_params_full())
        else:
            pardefs.extend(model_cls.trial_params_full())

        pardefs_dict = {}

        for pardef in pardefs:
            name_ = pardef['name']
            pardefs_dict[name_] = pardef

        return pardefs_dict
