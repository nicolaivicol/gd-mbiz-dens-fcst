import numpy as np
import pandas as pd
from tsfcst.models.abstract_model import TsModel


class MovingAverageModel(TsModel):

    def __init__(self, data, params, **kwargs):
        super().__init__(data, params)
        self.ma = None
        self.last_value = None

    def fit(self):
        n, avg = self.params['window'], self.params['average']
        x = pd.Series(self.data.target)
        n = min(n, len(x))

        if avg == 'simple':
            self.ma = x.rolling(n, min_periods=1).mean()
        elif avg == 'exponential':
            self.ma = x.ewm(span=n).mean()
        elif avg == 'weighted':
            weights = np.arange(1, n+1)
            self.ma = x.rolling(n).apply(lambda prices: np.dot(prices, weights) / weights.sum(), raw=True)
        else:
            raise ValueError(f'average type: {avg} not recognized. supported types: simple, exponential, weighted')

        self.last_value = self.ma[len(self.ma)-1]

    def _predict(self, steps):
        return np.array(np.repeat(self.last_value, steps))

    def _fitted_values(self):
        return np.array(self.ma)

    @staticmethod
    def default_params():
        return {
            'average': 'simple',
            'window': 12
        }

    @staticmethod
    def trial_params():
        params_trial = [
            dict(name='average', type='categorical', choices=['simple', 'exponential', 'weighted']),
            dict(name='window', type='int', low=1, high=24),
        ]
        return params_trial

    def flexibility(self):
        flexibility = 0
        avg_factor = {'simple': 1, 'weighted': 1.5, 'exponential': 2}[self.params['average']]
        flexibility += avg_factor * 1 / np.log1p(self.params['window'] / self.data.periods_year)
        return flexibility
