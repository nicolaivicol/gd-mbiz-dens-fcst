from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import polars as pl

from tsfcst.utils_tsfcst import update_nested_dict
from typing import List, Union, Dict
from tsfcst.time_series import TsData


class TsModel(ABC):

    def __init__(self, data: TsData, params: Dict, data_exog: List[TsData] = None):
        self.data = data
        self.data_exog = data_exog
        self.params = update_nested_dict(self.default_params(), params)

    @abstractmethod
    def fit(self, **kwargs) -> None:
        raise NotImplementedError

    def predict(self, steps) -> TsData:
        dates = self._time_range_predict(steps)
        values = self._predict(steps)
        ts_fcst = TsData(dates, values, freq=self.data.freq)
        return ts_fcst

    def fitted_values(self, as_polars=False) -> TsData:
        if as_polars:
            ts_fcst = pl.DataFrame({'date': self.data.time, 'value': self._fitted_values()})\
                .with_columns(pl.col('date').cast(pl.Date))
        else:
            ts_fcst = TsData(self.data.time, self._fitted_values(), freq=self.data.freq)
        return ts_fcst

    @abstractmethod
    def _predict(self, steps) -> Union[pd.Series, np.array]:
        raise NotImplementedError

    @abstractmethod
    def _fitted_values(self) -> Union[pd.Series, np.array]:
        """ Fitted values in-sample """
        raise NotImplementedError

    @staticmethod
    def default_params() -> Dict:
        return {}

    @staticmethod
    def trial_params(trend=True, seasonal=True, multiplicative=True, level=True, damp=False) -> List[Dict]:
        """
        Parameters search spaces (range/choices) to pick from in optuna trials
        These spaces can be narrowed conditional on prior beliefs about the series using parameters.

        Parameters:
            trend: possible to have trend? (True: adds one more option to consider, that trend can be also multiplicative)
            seasonal: possible to be seasonal?
            multiplicative: possible to be multiplicative?
            level: possible to have level changes / structural breaks?
            damp: to damp trend? (this will force one choice only: True or False)

        """
        raise NotImplementedError('trial_params() not implemented')

    @staticmethod
    def trial_params_full():
        """ That's to show the full possible search space, unrestricted """
        raise NotImplementedError('trial_params_full() not implemented')

    @staticmethod
    def names_params() -> List[str]:
        return [p['name'] for p in TsModel.trial_params_full()]

    def flexibility(self) -> float:
        """ Flexibility of the model. We can penalize flexibility to reduce overffiting. """
        raise NotImplementedError('flexibility() not implemented')

    def _time_range_predict(self, steps):
        forecast_start = np.max(self.data.time) + pd.DateOffset(**{self.data.interval_name: 1})
        dates = pd.date_range(forecast_start, periods=steps, freq=self.data.freq)
        return dates
