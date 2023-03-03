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
            method: str = 'average',
            weights: Dict[str, float] = None,
    ):
        self.data = data
        self.fcster_configs = fcster_configs
        self.forecasters = self._set_forecasters()
        self.method = self._set_method(method)
        self.weights = self._set_weights(weights, method)

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

        if self.method in ['single']:
            df_ens = pl.DataFrame({'ensemble': df_fcsts[self.names[0]]})
        elif self.method in ['weighted_average', 'wavg']:
            df_ens = pl.DataFrame({'ensemble': np.dot(df_fcsts.select(self.names).to_numpy(), self.weights_arr)})
        elif self.method in ['average', 'avg']:
            df_ens = pl.DataFrame({'ensemble': df_fcsts.select(self.names).mean(axis=1)})
        elif self.method in ['minimum', 'min']:
            df_ens = pl.DataFrame({'ensemble': df_fcsts.select(self.names).min(axis=1)})
        elif self.method in ['maximum', 'max']:
            df_ens = pl.DataFrame({'ensemble': df_fcsts.select(self.names).max(axis=1)})
        elif self.method == 'median':
            df_ens = df_fcsts \
                .select(self.names) \
                .with_columns(pl.concat_list(pl.all()).arr.eval(pl.all().median()).arr.get(0).alias('ensemble')) \
                .select('ensemble')
        else:
            raise ValueError(f'method={self.method} not recognized')

        df_fcsts = pl.concat([df_fcsts, df_ens], how='horizontal')

        return df_fcsts

    def _set_method(self, method: str):
        if method in ['single', 'weighted_average', 'wavg', 'average', 'avg',
                      'minimum', 'min', 'maximum', 'max', 'median']:
            if method == 'single':
                assert len(self.names) == 1
            return method
        else:
            raise ValueError(f'method={method} not recognized')

    def _set_weights(self, weights: Dict[str, float], method: str):
        if method == 'weighted_average':
            return {name_: weights[name_] for name_ in self.fcster_configs.keys()}
        else:
            return {name_: np.NaN for name_ in self.fcster_configs.keys()}

    def _set_forecasters(self) -> Dict[str, Forecaster]:
        return {name: Forecaster.from_config(data=self.data, cfg=cfg) for name, cfg in self.fcster_configs.items()}


