import pandas as pd
import numpy as np
import math
from typing import List
from tsfcst.forecasters.forecaster import Forecaster
from tsfcst.time_series import TsData
from tsfcst.forecasters.inventory import FORECASTERS, DEFAULT_FALLBACK
# from tsfcst.model_error.utils import add_derived_feats
# from tsfcst.model_error.inventory import MODELS_ERROR, MODELS_WEIGHT
# from tsfcst.utils import get_feats


class ConfigSelector:

    def __init__(
            self,
            data: TsData,
            models: list = None,
            fallback: str = None,
            model_error_name: str = 'default',
            model_error_weight: float = None,
    ):
        self.data = data
        # self.model_error_weight = (model_error_weight
        #                            if model_error_weight is not None
        #                            else MODELS_WEIGHT[model_error_name])
        # self.model_error = MODELS_ERROR.get(model_error_name, None)
        self.models = self._set_models(models)
        self.fallback = fallback if fallback is not None else DEFAULT_FALLBACK
        self.cv_res = None
        self.best_config = None
        self.best_metric = None
        self.feats = None

    def select(
            self,
            n_train_dates=3,
            step_train_dates=2,
            periods_ahead=5,
            min_train_size=24,
            metric='smape_avg',
    ) -> str:

        possible = self.models.copy()

        if len(possible) > 1:
            possible = self._filter_out_unfit_configs(possible)

        if len(possible) > 1:
            possible = self._select_by_cv(possible, n_train_dates, step_train_dates, periods_ahead, min_train_size, metric)

        # if len(possible) > 1 and self.use_model_for_error() and self.cv_res is not None:
        #     possible = self._select_by_model(metric)

        self.best_config = possible[0] if len(possible) > 0 else self.fallback

        return self.best_config

    def _set_models(self, models):
        if models is None:
            models = list(FORECASTERS.keys())
        self.models = [m for m in models if m in list(FORECASTERS.keys())]
        return self.models

    def _filter_out_unfit_configs(self, posible_configs):
        not_good = []
        has_short_history = len(self.data.data) < ((2*53 + 1)*7)

        for config_name, config in FORECASTERS.items():
            freq_model = config.get('freq_model', 'D')
            if freq_model == 'W' and has_short_history:
                not_good.append(config_name)
            elif freq_model == 'D' and not has_short_history:
                not_good.append(config_name)

        posible_configs = [c for c in posible_configs if c not in not_good]
        return posible_configs

    # def use_model_for_error(self):
    #     return self.model_error is not None and self.model_error_weight > 0

    def _select_by_cv(self, possible, train_dates_n, train_dates_freq, periods_ahead, min_train_size, metric):
        cv_res = []
        for config_name in possible:
            try:
                fcster = Forecaster.from_named_configs(data=self.data, config_name=config_name)
                n_data_points_free = len(self.data.data) - min_train_size - periods_ahead
                if n_data_points_free < 0:
                    raise AssertionError('Not enough data for validation.')
                n_train_dates_possible = max(1, math.floor(n_data_points_free / train_dates_freq))
                train_dates_n = min(train_dates_n, n_train_dates_possible)
                df_fcsts_cv, metrics_cv = fcster.cv(train_dates_n, train_dates_freq, periods_ahead)
                metrics_cv['config_name'] = config_name
                cv_res.append(metrics_cv)
            except Exception as e:
                print(f'CV failed for config={config_name}, with error: {str(e)}')
                pass

        if len(cv_res) == 0:
            print(f'cv failed for all possible configs, returning fallback option: {self.fallback}')
            return [self.fallback]

        cv_res = pd.DataFrame(cv_res)
        cv_res.sort_values(by=metric, ascending=True, inplace=True, ignore_index=True)
        self.cv_res = cv_res
        self.best_config, self.best_metric = cv_res.loc[0, 'config_name'], round(cv_res.loc[0, metric], 3)
        return list(self.cv_res['config_name'])

    # def _select_by_model(self, metric):
    #     self.feats = compute_features(self.data.target, self)
    #     X = self.feats.copy()
    #     feature_names = self.model_error.feature_name()
    #     for f in feature_names:
    #         if f not in self.feats.columns:
    #             X[f] = np.NaN
    #     X = X[feature_names]
    #     self.cv_res[f'pred_{metric}'] = self.model_error.predict(X.values)
    #
    #     w = self.model_error_weight
    #     self.cv_res[f'weighted_metric'] = (1 - w) * self.cv_res[metric] + w * self.cv_res[f'pred_{metric}']
    #     self.cv_res.sort_values(by='weighted_metric', ascending=True, inplace=True, ignore_index=True)
    #     self.best_config, self.best_metric = self.cv_res.loc[0, 'config_name'], round(self.cv_res.loc[0, 'weighted_metric'], 3)
    #     return list(self.cv_res['config_name'])


# def compute_features(
#         values,
#         config_selector: ConfigSelector = None,
#         add_feats_series=True,
#         add_cv_features=True,
#         add_derived_feats_=True,
# ):
#     feats = {}
#
#     if add_feats_series:
#         feats_series = get_feats(values)
#         feats = {**feats, **feats_series}
#
#     if config_selector is not None and add_cv_features:
#         feats_from_cv = {}
#         feats_from_cv['slct_mape_30'] = config_selector.best_metric
#         feats_from_cv['slct_model'] = config_selector.best_config
#         for i, row in config_selector.cv_res.iterrows():
#             cnfg = row['config_name']
#             feats_from_cv[f'mape_30_{cnfg}'] = row['mape_30']
#             feats_from_cv[f'error_{cnfg}'] = row['error_totals']
#         feats = {**feats, **feats_from_cv}
#
#     for k, v in feats.items():
#         try:
#             feats[k] = max(-99999, min(99999, v))
#         except:
#             pass
#
#     df_feats = pd.DataFrame([feats])
#
#     if add_derived_feats_:
#         n_configs = len(config_selector.cv_res)
#         df_feats = pd.DataFrame(np.repeat(df_feats.values, n_configs, axis=0), columns=df_feats.columns)
#         df_feats = pd.concat([df_feats, config_selector.cv_res], axis=1)
#         df_feats = add_derived_feats(df_feats)
#
#     return df_feats
#

def run_cv(
        data: TsData,
        configs: List[str],
        train_dates_n: int,
        train_dates_freq: int,
        periods_ahead: int,
        min_train_size: int,
):
    list_metrics_cv = []
    list_df_fcsts_cv = []

    for config_name in configs:
        try:
            fcster = Forecaster.from_named_configs(data=data, config_name=config_name)
            n_data_points_free = len(data) - min_train_size - periods_ahead

            if n_data_points_free < 0:
                raise AssertionError('Not enough data for validation.')

            n_train_dates_possible = max(1, math.floor(n_data_points_free / train_dates_freq))
            train_dates_n = min(train_dates_n, n_train_dates_possible)
            df_fcsts_cv, metrics_cv = fcster.cv(train_dates_n, train_dates_freq, periods_ahead)
            metrics_cv['config_name'] = config_name
            df_fcsts_cv['config_name'] = config_name
            list_metrics_cv.append(metrics_cv)
            list_df_fcsts_cv.append(df_fcsts_cv)

        except Exception as e:
            print(f'CV failed for config={config_name}, with error: {str(e)}')
            pass

