import math
import numpy as np
import pandas as pd
import warnings
from typing import List, Dict, Union
import copy
from scipy.special import boxcox1p, inv_boxcox1p
# from sklearn.preprocessing import MinMaxScaler

from tsfcst.forecasters.inventory import FORECASTERS
from tsfcst.time_series import TsData
from tsfcst.models.abstract_model import TsModel
from tsfcst.models.inventory import MODELS
from tsfcst.utils import (
    calc_fcst_error_metrics,
    treat_outliers,
    smooth_series,
    limit_min_max
)

warnings.simplefilter(action='ignore')
_EPSILON = 1e-4


class Forecaster:

    def __init__(
            self,
            model_cls: type(TsModel),
            data: Union[pd.DataFrame, TsData],
            data_exog: List[Union[pd.DataFrame, TsData]] = None,
            target_name: str = 'value',
            forecast_name: str = 'fcst',
            time_name: str = 'date',
            feature_names: list = None,
            freq_data: str = 'MS',
            freq_model: str = 'MS',
            params_model: dict = None,
            outliers: bool = False,
            params_outliers: dict = None,
            smooth: bool = False,
            params_smooth: dict = None,
            log_transform: bool = False,
            sub_zero_with_eps: bool = False,
            fcst_min_max: bool = False,
            params_fcst_min_max: dict = None,
            boxcox_lambda: float = None,  # 1: is equivalent to using the original data, 0: log transform
            normalize: float = False,
            use_data_since: str = None,
            **kwargs,
    ):
        self.model_cls = model_cls  # model class, must be a TsModel
        self.data = set_data(data, time_name, target_name, freq_data)
        if use_data_since is not None and use_data_since != 'all':
            self.data.data = self.data.data.loc[self.data.time >= pd.to_datetime(use_data_since)].reset_index(drop=True)
        self.data_exog = []  # set_data_list(data_exog, time_name, target_name, freq_data)
        self.target_name = target_name
        self.forecast_name = forecast_name
        self.time_name = time_name
        self.feature_names = feature_names if feature_names is not None else []
        self.freq_data = freq_data
        self.freq_model = freq_model

        self.params_model = params_model if params_model is not None else {}

        # data transformation before train
        self.params = {}
        self.params['outliers'] = outliers
        self.params['params_outliers'] = params_outliers if params_outliers is not None else {}
        self.params['smooth'] = smooth
        self.params['params_smooth'] = params_smooth if params_smooth is not None else {}
        self.params['log_transform'] = self._set_log_transform(log_transform)
        self.params['sub_zero_with_eps'] = sub_zero_with_eps
        self.params['boxcox_lambda'] = boxcox_lambda
        self.params['normalize'] = normalize
        self.params['use_data_since'] = use_data_since

        # forecast transformation after train
        self.params['fcst_min_max'] = fcst_min_max
        self.params['params_fcst_min_max'] = params_fcst_min_max if params_fcst_min_max is not None else {}

        # set by fit()
        self._data_train = None     # data used for last fit()
        self._data_exog_train = None
        self._scaler = None         # min-max scaler
        self.model: TsModel = None  # fit model

        # set by cv()
        self.df_fcsts_cv = None
        self.metrics_cv = None

    @classmethod
    def from_named_configs(cls, data: Union[pd.DataFrame, TsData], config_name: str, **kwargs) -> 'Forecaster':
        config = copy.deepcopy(FORECASTERS[config_name])
        config.update(kwargs)
        return cls(model_cls=MODELS[config['model_name']], data=data, **config)

    @staticmethod
    def trial_params() -> List[Dict]:
        trial_params_ = [
            # dict(name='boxcox_lambda', type='float', low=0.0, high=1.0),
            dict(name='use_data_since', type='categorical', choices=['all', '2020-02-01', '2021-02-01']) #
        ]
        return trial_params_

    def fit(self, train_date=None):
        self._data_train = self._prep_data_train(train_date=train_date)
        self._data_exog_train = self._prep_data_exog_train(train_date=train_date)
        self.model = self.model_cls(data=self._data_train, params=self.params_model, data_exog=self._data_exog_train)
        self.model.fit()

    def forecast(self, periods_ahead: int = None, train_date=None, forecast_end=None) -> pd.DataFrame:
        periods_ahead, train_date, forecast_end = self._prep_args_predict(periods_ahead, train_date, forecast_end, self.freq_model)
        self.fit(train_date=train_date)
        ts_pred = self.model.predict(steps=periods_ahead)
        df_fcst = self._prep_forecast_df(ts_pred, train_date, forecast_end)
        return df_fcst

    def cv(self, n_train_dates=None, step_train_dates=None, periods_val=None,
           periods_test=0, periods_out=0, periods_val_last=None):

        n_train_dates, step_train_dates, periods_val, periods_val_last = \
            self._handle_cv_args(n_train_dates, step_train_dates, periods_val, periods_val_last)
        train_dates, test_start, last_date, periods_fcst = \
            self._handle_cv_dates(n_train_dates, step_train_dates, periods_val, periods_test, periods_out, periods_val_last)

        t, v, f = self.time_name, self.target_name, self.forecast_name
        df_fcsts_cv, df_fits_cv = pd.DataFrame({t: []}), pd.DataFrame({t: []})

        for td, p in zip(train_dates, periods_fcst):
            # keeps forecasts on validation / test / out
            df_fcst = self.forecast(train_date=td, periods_ahead=p)
            df_fcst.rename(columns={f: f"{f}_{td.strftime('%Y%m%d')}"}, inplace=True)
            df_fcsts_cv = pd.merge(df_fcsts_cv, df_fcst, how='outer', on=[self.time_name])
            # keep fitted values
            df_fit = self.model.fitted_values().data
            df_fit.rename(columns={'value': f"fit_{td.strftime('%Y%m%d')}"}, inplace=True)
            df_fit = df_fit.tail(periods_val)
            df_fits_cv = pd.merge(df_fits_cv, df_fit, how='outer', on=[self.time_name])

        df_fcsts_cv.sort_values(by=t, inplace=True)
        df_fits_cv.sort_values(by=t, inplace=True)

        metrics_cv = {}

        # calculate metrics on the validation folds (out-of-sample)
        cols_valid = [f"{f}_{td.strftime('%Y%m%d')}" for td in train_dates if td not in [test_start, last_date]]
        metrics_val = calc_fcst_error_metrics(self.data.data[[t, v]], df_fcsts_cv[[t] + cols_valid], time_name=t, target_name=v)
        metrics_val = {k + '_val': v for k, v in metrics_val.items()}
        metrics_cv.update(metrics_val)

        # calculate metrics on the test fold (out-of-sample)
        if periods_test > 0:
            cols_test = [t] + [f"{f}_{test_start.strftime('%Y%m%d')}"]
            metrics_test = calc_fcst_error_metrics(self.data.data[[t, v]], df_fcsts_cv[cols_test], time_name=t, target_name=v)
            metrics_test = {k + '_test': v for k, v in metrics_test.items()}
            metrics_cv.update(metrics_test)

        # calculate metrics on the fitted values (in-sample)
        cols_fit = [t] + [f"fit_{td.strftime('%Y%m%d')}" for td in train_dates if td not in [test_start, last_date]]
        metrics_fit = calc_fcst_error_metrics(self.data.data[[t, v]], df_fits_cv[cols_fit], time_name=t, target_name=v)
        metrics_fit = {k + '_fit': v for k, v in metrics_fit.items()}
        metrics_cv.update(metrics_fit)

        self.df_fcsts_cv, self.metrics_cv = df_fcsts_cv, metrics_cv
        return df_fcsts_cv, metrics_cv

    def _handle_cv_args(self, n_train_dates, step_train_dates, periods_val, periods_val_last):
        if n_train_dates is None and step_train_dates is None and periods_val is None and periods_val_last is None:
            defaults = {'D': (5, 28, 28, 28), 'W': (5, 4, 4, 5), 'M': (3, 2, 7, 5), 'MS': (3, 2, 7, 5)}
            n_train_dates, step_train_dates, periods_val, periods_val_last = defaults[self.freq_model]
        return n_train_dates, step_train_dates, periods_val, periods_val_last

    def _handle_cv_dates(self, n_train_dates, step_train_dates, periods_val, periods_test, periods_out, periods_val_last):
        target_not_na = ~np.isnan(self.data.target)
        last_date = pd.to_datetime(np.max(self.data.time[target_not_na]))
        intrvl_name = TsData.freq_to_interval_name(self.freq_model)
        test_start = last_date - pd.DateOffset(**{intrvl_name: periods_test})
        min_req_data_obs = self._min_required_obs(self.freq_model)
        min_allowed_train_date = min(self.data.time) + pd.DateOffset(**{intrvl_name: min_req_data_obs})

        if periods_val_last is None:
            periods_val_last = periods_val

        train_dates, periods_fcst = [], []

        for i in sorted(range(n_train_dates), reverse=True):
            periods_offset = periods_val_last + i * step_train_dates
            train_date = test_start - pd.DateOffset(**{intrvl_name: periods_offset})
            if train_date < min_allowed_train_date:
                raise AssertionError(f"{min_allowed_train_date.strftime('%Y-%m-%d')} is the minimum train date which "
                                     f"ensures at least {min_req_data_obs} data points for training, but the provided "
                                     f"settings generated the train date {train_date.strftime('%Y-%m-%d')} "
                                     f"which is before this date, use less n_train_dates or smaller step_train_dates")
            train_dates.append(train_date)
            periods_fcst.append(min(periods_val, periods_offset))

        if periods_test > 0:
            train_dates.append(test_start)
            periods_fcst.append(max(periods_val, periods_test))

        if periods_out > 0:
            train_dates.append(last_date)
            periods_fcst.append(periods_out)

        return train_dates, test_start, last_date, periods_fcst

    @staticmethod
    def _min_required_obs(freq) -> int:
        """ minimum required data points to fit a model """
        defaults = {'D': 180, 'W': 52, 'M': 5, 'MS': 5}
        return defaults[freq]

    @staticmethod
    def metrics_cv_str_pretty(metrics_cv):
        s = (f"smape_fit={metrics_cv['smape_avg_fit']:.3f}±{metrics_cv['smape_std_fit']:.3f}"
             f" | smape_val={metrics_cv['smape_avg_val']:.3f}±{metrics_cv['smape_std_val']:.3f}"
             f" | smape_test={metrics_cv.get('smape_avg_test', np.nan):.3f}"
             f" | maprev_val={metrics_cv['maprev_val']:.3f}"
             f" | irreg_val={metrics_cv['irreg_val']:.3f}")
        return s

    def _prep_forecast_df(self, ts_fcst: TsData, train_date=None, forecast_end=None):
        v = ts_fcst.name_value

        if self.params['boxcox_lambda'] is not None:
            ts_fcst.data[v] = inv_boxcox1p(ts_fcst.data[v], self.params['boxcox_lambda'])
        elif self.params['log_transform']:
            ts_fcst.data[v] = np.expm1(ts_fcst.data[v])

        # if self.params['normalize']:
        #     ts_fcst.data[v] = self._scaler.inverse_transform(ts_fcst.data[v])

        if self.freq_data != self.freq_model:
            ts_fcst = ts_fcst.disaggregate(to_freq=self.freq_data)

        if self.params['fcst_min_max']:
            if train_date is not None:
                y_train = self.data.target[self.data.time <= pd.to_datetime(train_date)]
            else:
                y_train = self.data.target

            if self.freq_data == 'D':
                y_train = y_train.rolling(7, min_periods=1).mean()

            ts_fcst.data[v] = limit_min_max(
                y=ts_fcst.data[v],
                y_train=y_train,
                **self.params['params_fcst_min_max']
            )

        if forecast_end is not None:
            ts_fcst.data = ts_fcst.data.loc[ts_fcst.time <= pd.to_datetime(forecast_end), ]

        if train_date is not None:
            ts_fcst.data = ts_fcst.data.loc[pd.to_datetime(train_date) < ts_fcst.time, ]

        df_fcst = pd.DataFrame({
            self.time_name: np.array(ts_fcst.time),
            self.forecast_name: np.round(np.array(ts_fcst.target), 5)
        })

        return df_fcst

    def _prep_args_predict(self, periods_ahead, train_date, forecast_end, freq='D'):
        latest_available_date = max(self.data.time)
        if train_date is None:
            train_date = latest_available_date
        else:
            train_date = min(pd.to_datetime(train_date), latest_available_date)

        if periods_ahead is None:
            if train_date is not None and forecast_end is not None:
                periods_ahead = int(forecast_end - train_date)
            else:
                periods_ahead = 1

        offset_ = {
            'D': pd.DateOffset(days=periods_ahead),
            'W': pd.DateOffset(weeks=periods_ahead),
            'M': pd.DateOffset(months=periods_ahead),
            'MS': pd.DateOffset(months=periods_ahead),
            'Q': pd.DateOffset(months=periods_ahead*3),
            'QS': pd.DateOffset(months=periods_ahead*3),
            'Y': pd.DateOffset(years=periods_ahead),
        }
        forecast_end = train_date + offset_.get(freq, pd.DateOffset(periods_ahead))

        if self.freq_data != self.freq_model:
            num_freq_data = float(TsData.freq_numeric(self.freq_data))
            num_freq_model = float(TsData.freq_numeric(self.freq_model))
            periods_ahead = math.ceil(periods_ahead * num_freq_data/num_freq_model)

        return periods_ahead, train_date, forecast_end

    def _prep_data_exog_train(self, train_date=None):
        if self.data_exog is None:
            return None

        l_ts = []
        for ts in self.data_exog:
            ts = ts.deepcopy()

            if train_date is not None:
                ts.data = ts.data.loc[ts.time <= pd.to_datetime(train_date)]

            if self.freq_data != self.freq_model:
                ts = ts.aggregate(to_freq=self.freq_model)

            l_ts.append(ts)

        return l_ts

    def _prep_data_train(self, train_date=None):
        ts = self.data.deepcopy()

        if train_date is not None:
            ts.data = ts.data.loc[ts.time <= pd.to_datetime(train_date)]

        if self.freq_data != self.freq_model:
            ts = ts.aggregate(to_freq=self.freq_model)

        # if self.params['normalize']:
        #     values = ts.data[ts.name_value].values
        #     values = values.reshape((len(values), 1))
        #     self._scaler = MinMaxScaler(feature_range=(0, 1))
        #     self._scaler.fit(values)
        #     ts.data[ts.name_value] = self._scaler.transform(ts.data[ts.name_value].values)

        if self.params['boxcox_lambda'] is not None:
            ts.data[ts.name_value] = boxcox1p(ts.data[ts.name_value], self.params['boxcox_lambda'])
        elif self.params['log_transform']:
            ts.data[ts.name_value] = np.log1p(ts.data[ts.name_value])

        if self.params['outliers']:
            ts.data[ts.name_value] = treat_outliers(
                y=ts.data[ts.name_value],
                freq=ts.freq,
                **self.params['params_outliers']
            )

        if self.params['smooth']:
            ts.data[ts.name_value] = smooth_series(
                y=ts.data[ts.name_value],
                **self.params['params_smooth']
            )

        if self.params['sub_zero_with_eps']:
            is_zero = ts.target == 0
            if any(is_zero):
                n_zeros = np.sum(is_zero * 1)
                small_value = _EPSILON + _EPSILON * np.median(ts.target)
                small_values = small_value + np.random.random(n_zeros) * small_value
                ts.data.loc[is_zero, ts.name_value] = small_values

        return ts

    def _set_log_transform(self, log_transform):
        self.params['log_transform'] = log_transform
        if log_transform and any(self.data.target < 0):
            self.params['log_transform'] = False
            raise Warning('log1p transformation cannot be applied to series having negative values,'
                          ' setting log_transform=False')
        return self.params['log_transform']


def set_data(data: Union[pd.DataFrame, TsData], time_name, target_name, freq_data) -> TsData:
    if isinstance(data, pd.DataFrame):
        dates, values = data[time_name], data[target_name]
    elif isinstance(data, TsData):
        dates, values = data.time, data.target
    else:
        raise ValueError('data can be only of types: pd.DataFrame, TsData')

    data = TsData(
        dates=dates,
        values=values,
        name_date=time_name,
        name_value=target_name,
        freq=freq_data,
    )
    return data


def set_data_list(list_data: List[Union[pd.DataFrame, TsData]], time_name, target_name, freq_data) -> List[TsData]:
    return [set_data(data, time_name, target_name, freq_data) for data in list_data]


