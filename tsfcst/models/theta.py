import numpy as np
from tsfcst.models.abstract_model import TsModel
from statsmodels.tsa.forecasting.theta import ThetaModel, ThetaModelResults


class ThetaSmModel(TsModel):
    """
    Theta model with statsmodels implementation

    https://www.statsmodels.org/dev/examples/notebooks/generated/theta-model.html
    https://www.statsmodels.org/stable/generated/statsmodels.tsa.forecasting.theta.ThetaModel.html#statsmodels.tsa.forecasting.theta.ThetaModel

    Parameters
    theta:
        larger theta allow for stronger trend, lower thetas produce flatter forecasts; range: 1 and higher
    """

    def __init__(self, data, params, **kwargs):
        super().__init__(data, params)
        self.model: ThetaModel = None
        self.model_fit: ThetaModelResults = None

    def fit(self):
        self.model = ThetaModel(
            endog=self.data.target,
            period=self.params['period'],
            deseasonalize=self.params['deseasonalize'],
            use_test=self.params['use_test'],
            method=self.params['method'],
        )
        self.model_fit = self.model.fit(use_mle=self.params['use_mle'])

    def _predict(self, steps):
        return self.model_fit.forecast(steps, theta=self.params['theta'])

    def _fitted_values(self):
        return np.array(self.data.target)  # Theta has no fit values in sample

    @staticmethod
    def default_params():
        return {
            'period': 12,
            'deseasonalize': False,
            'use_test': True,
            'method': 'additive',
            'theta': 2.0,
            'use_mle': False,
        }

    @staticmethod
    def trial_params(trend=True, seasonal=True, multiplicative=True, level=True, damp=False):

        theta_max = 5 if trend else 1.0001
        params_trial = [dict(name='theta', type='float', low=1.0, high=theta_max)]

        if seasonal:
            params_trial.append(dict(name='deseasonalize', type='categorical', choices=[True, False]))
            method_choices = ['additive']
            if multiplicative:
                method_choices.append('multiplicative')
            params_trial.append(dict(name='method', type='categorical', choices=method_choices))

        return params_trial

    def flexibility(self):
        use_mle = float(self.params['use_mle'])
        flexibility = (1 + use_mle) * (self.params['theta'] - 0.99) ** 2
        flexibility += float(self.params['deseasonalize']) * (1 + 1 * (self.params['method'] == 'multiplicative'))
        return flexibility
