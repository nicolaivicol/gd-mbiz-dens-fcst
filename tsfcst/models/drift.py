import numpy as np
from tsfcst.models.abstract_model import TsModel
from tsfcst.utils_tsfcst import last_n_from_x, rate_diffs


class DriftModel(TsModel):
    """
    Drift model: last observation with growth rate implied from history
    """

    def __init__(self, data, params, **kwargs):
        super().__init__(data, params)
        self.last_value = None
        self.hist_rate = None

    def fit(self):
        self.last_value = self.data.target[len(self.data.target)-1]
        if self.params['window_drift'] <= 1 or self.params['mult_drift'] <= 0.01:
            self.hist_rate = 0
        else:
            last_n_obs = last_n_from_x(self.data.target)
            self.hist_rate = np.nanmean(rate_diffs(last_n_obs))
            if np.isnan(self.hist_rate):
                self.hist_rate = 0

    def _predict(self, steps):
        mults = np.array([1 + self.hist_rate * n for n in range(1, steps+1)])  # additive model
        print(mults)
        return self.last_value * mults

    def _fitted_values(self):
        return np.array(self.data.target)

    @staticmethod
    def default_params():
        return {
            'window_drift': 18,
            'mult_drift': 1,
        }

    @staticmethod
    def trial_params(trend=None, seasonal=None, multiplicative=None, level=None, damp=None):
        max_window = 18 if trend else 1
        mult_max = 1.0 if trend else 0.01
        params_trial = [
            dict(name='window_drift', type='int', low=6, high=max_window, step=6),
            dict(name='mult_drift', type='float', low=0.0, high=mult_max, step=0.10),
        ]
        return params_trial

    @staticmethod
    def trial_params_full():
        return [
            dict(name='window_drift', type='int', low=0, high=18),
            dict(name='mult_drift', type='float', low=0.0, high=1, step=0.01),
        ]

    @staticmethod
    def trial_params_grid(trend=True, seasonal=True, multiplicative=True, level=True, damp=False):
        trial_params_ = DriftModel.trial_params(trend, seasonal, multiplicative, level, damp)
        return TsModel.trial_params_grid(trial_params_)

    @staticmethod
    def names_params():
        return [p['name'] for p in DriftModel.trial_params_full()]

    def flexibility(self):
        return 0
