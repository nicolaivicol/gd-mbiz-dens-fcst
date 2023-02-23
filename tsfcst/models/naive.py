import numpy as np
from tsfcst.models.abstract_model import TsModel


class NaiveModel(TsModel):
    """
    Naive model: last observation carried forward
    """

    def __init__(self, data, params, **kwargs):
        super().__init__(data, params)
        self.last_value = None

    def fit(self):
        self.last_value = self.data.target[len(self.data.target)-1]

    def _predict(self, steps):
        return np.array(np.repeat(self.last_value, steps))

    def _fitted_values(self):
        return np.array(self.data.target)

    @staticmethod
    def trial_params(trend=None, seasonal=None, multiplicative=None, level=None, damp=None):
        # does not depend on any parameters, this misc param is to have a non-empty list
        return [dict(name='misc_naive', type='categorical', choices=['last_obs'])]

    @staticmethod
    def trial_params_full():
        return NaiveModel.trial_params()

    @staticmethod
    def trial_params_grid(trend=True, seasonal=True, multiplicative=True, level=True, damp=False):
        trial_params_ = NaiveModel.trial_params(trend, seasonal, multiplicative, level, damp)
        return TsModel.trial_params_grid(trial_params_)

    @staticmethod
    def names_params():
        return [p['name'] for p in NaiveModel.trial_params_full()]

    def flexibility(self):
        return 0
