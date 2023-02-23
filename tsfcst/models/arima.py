import numpy as np
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults
from tsfcst.models.abstract_model import TsModel


class ArimaModel(TsModel):
    """
    ARIMA
    https://www.statsmodels.org/dev/generated/statsmodels.tsa.arima.model.ARIMA.html
    """

    def __init__(self, data, params, **kwargs):
        super().__init__(data, params)
        self.model: ARIMA = None
        self.model_fit: ARIMAResults = None

    def fit(self):
        trend_ = self.params['trend'] if self.params['trend'] in ['t', 'ct'] else None
        self.model = ARIMA(
            endog=self.data.target,
            order=(self.params['p'], self.params['d'], self.params['q']),
            trend=trend_,
        )
        self.model_fit = self.model.fit()

    def _predict(self, steps):
        return self.model_fit.forecast(steps)

    def _fitted_values(self):
        return self.model_fit.fittedvalues

    @staticmethod
    def default_params():
        return {'p': 0, 'd': 1, 'q': 0, 'trend': None}

    @staticmethod
    def trial_params(trend=True, seasonal=True, multiplicative=True, level=True, damp=False):
        params_trial = [
            dict(name='p', type='int', low=0, high=3),
            dict(name='d', type='int', low=0, high=1),
            dict(name='q', type='int', low=0, high=3),
        ]
        if trend:
            params_trial.append(dict(name='trend', type='categorical', choices=['no', 't']))
        return params_trial

    @staticmethod
    def trial_params_full():
        return ArimaModel.trial_params()

    @staticmethod
    def trial_params_grid(trend=True, seasonal=True, multiplicative=True, level=True, damp=False):
        trial_params_ = ArimaModel.trial_params(trend, seasonal, multiplicative, level, damp)
        return TsModel.trial_params_grid(trial_params_)

    @staticmethod
    def names_params():
        return [p['name'] for p in ArimaModel.trial_params_full()]

    def flexibility(self):
        k = self.params['p'] + self.params['d'] + self.params['q']
        t = 0.5 * (self.params.get('t', 'no') == 't') * k
        n_obs = len(self.data)
        return (k + t) - 0.5 * np.log(n_obs)
