from typing import List, Dict
import polars as pl
import numpy as np

from tsfcst.time_series import TsData
from tsfcst.forecasters.forecaster import Forecaster, ForecasterConfig


class Ensemble:

    def __init__(
            self,
            data: TsData,
            fcster_configs: Dict[str, ForecasterConfig],
            weights: Dict[str, float],
    ):
        self.data = data
        self.fcster_configs = fcster_configs
        self.forecasters = self._set_forecasters()
        self.weights = self._set_weights(weights)

    @property
    def weights_arr(self) -> np.ndarray:
        return np.array(list(self.weights.values()))

    @property
    def names(self) -> List[str]:
        return list(self.forecasters.keys())

    def forecast(self, periods_ahead: int) -> pl.DataFrame:
        fcsts = {}

        for name in self.names:
            forecaster = self.forecasters[name]
            fcst = forecaster.forecast(periods_ahead, as_ts=True)
            fcsts['date'] = fcst.time
            fcsts[name] = fcst.target * 1.0

        df_fcsts = pl.DataFrame(fcsts).with_columns(pl.col('date').cast(pl.Date))
        df_ens = pl.DataFrame({'ensemble': np.dot(df_fcsts.select(self.names).to_numpy(), self.weights_arr)})
        df_fcsts = pl.concat([df_fcsts, df_ens], how='horizontal')

        return df_fcsts

    def _set_weights(self, weights: Dict[str, float]):
        return {name_: weights[name_] for name_ in self.fcster_configs.keys()}

    def _set_forecasters(self) -> Dict[str, Forecaster]:
        return {name: Forecaster.from_config(data=self.data, cfg=cfg) for name, cfg in self.fcster_configs.items()}


