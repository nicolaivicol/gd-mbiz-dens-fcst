import numpy as np
import pmdarima
from tsfcst.models.abstract_model import TsModel


class ArimaAutoModel(TsModel):
    """
    Auto ARIMA
    https://github.com/alkaline-ml/pmdarima
    """

    def __init__(self, data, params, **kwargs):
        super().__init__(data, params)
        self.model = None

    def fit(self):
        self.model = pmdarima.auto_arima(
            self.data.target,
            **self.params,
            # trace=True,
            error_action='ignore',
            suppress_warnings=True,
        )

    def _predict(self, steps):
        return self.model.predict(steps)

    def _fitted_values(self):
        return np.array(self.model.predict_in_sample())

    @staticmethod
    def default_params():
        return {
            'start_p': 1,
            'start_q': 1,
            'max_p': 3,
            'max_q': 3,
            'max_d': 2,
            'm': 1,
            'seasonal': False,
        }

    @staticmethod
    def trial_params(trend=True, seasonal=True, multiplicative=True, level=True, damp=False):
        params_trial = [
            dict(name='max_p', type='int', low=1, high=3),
            dict(name='max_q', type='int', low=1, high=3),
            dict(name='max_d', type='int', low=1, high=2),
        ]
        return params_trial

    @staticmethod
    def trial_params_full():
        return ArimaAutoModel.trial_params()

    @staticmethod
    def names_params():
        return [p['name'] for p in ArimaAutoModel.trial_params_full()]

    def flexibility(self):
        return 0
