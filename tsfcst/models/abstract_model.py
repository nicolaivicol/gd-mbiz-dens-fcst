from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from tsfcst.utils import update_nested_dict
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

    def fitted_values(self) -> TsData:
        ts_fcst = TsData(self.data.time, self._fitted_values(), freq=self.data.freq)
        return ts_fcst

    @abstractmethod
    def _predict(self, steps) -> Union[pd.Series, np.array]:
        raise NotImplementedError

    @abstractmethod
    def _fitted_values(self) -> Union[pd.Series, np.array]:
        raise NotImplementedError

    @staticmethod
    def default_params() -> Dict:
        return {}

    @staticmethod
    def trial_params() -> List[Dict]:
        """ Parameters range/choices to try with optuna. """
        raise NotImplementedError('implement trial_params()')

    def _time_range_predict(self, steps):
        forecast_start = np.max(self.data.time) + pd.DateOffset(**{self.data.interval_name: 1})
        dates = pd.date_range(forecast_start, periods=steps, freq=self.data.freq)
        return dates
