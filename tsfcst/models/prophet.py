import numpy as np
from tsfcst.models.abstract_model import TsModel
from prophet import Prophet
import logging


# suppress logs by `cmdstanpy`
log_cmdstanpy = logging.getLogger('cmdstanpy')
log_cmdstanpy.addHandler(logging.NullHandler())
log_cmdstanpy.propagate = False
log_cmdstanpy.setLevel(logging.INFO)


class ProphetModel(TsModel):
    """
    Prophet model
    https://facebook.github.io/prophet/

    Parameters
    growth:
        "linear" (default): the trend will grow at a constant rate
        "logistic": the growth rate will start high and then taper off over time.
        "flat": no trend
    n_changepoints:
        This parameter specifies the number of changepoints to include in the trend component of the model.
        A changepoint represents a time when the rate of growth in the time series changes.
        A higher value of n_changepoints will result in a more flexible model that can capture more complex trends,
        but may also result in overfitting.
    changepoint_range:
        Proportion of history in which trend changepoints will be estimated. Defaults to 0.8 for the first 80% of data.
        Use lower values to avoid overfitting fluctuations at the end of the time series.
        Higher values makes the model sensitive to the data at the end of the series.
    changepoint_prior_scale:
        Parameter modulating the flexibility of the automatic changepoint selection.
        Large values will allow many changepoints, small values will allow few changepoints.
    seasonality_mode:
        ‘additive’ (default) or ‘multiplicative’.
    seasonality_prior_scale: Parameter modulating the strength of the seasonality model.
        Larger values allow the model to fit larger seasonal fluctuations, smaller values dampen the seasonality.
    """

    def __init__(self, data, params, data_exog=None):
        super().__init__(data, params, data_exog)
        self.model = None
        self.data_train = self.data.data.rename(columns={self.data.name_date: 'ds', self.data.name_value: 'y'})

    def fit(self):
        self.model = Prophet(**self.params)
        self.model.fit(self.data_train)

    def _predict(self, steps):
        df_future = self.model.make_future_dataframe(periods=steps, include_history=False, freq='MS')
        df_pred = self.model.predict(df_future)
        return np.array(df_pred['yhat'])

    def _fitted_values(self):
        fcst_insample = self.model.predict(self.data_train)
        return np.array(fcst_insample['yhat'].values)

    @staticmethod
    def default_params():
        return {
            'growth': 'linear',
            'n_changepoints': 3,
            'changepoint_range': 0.95,
            'changepoint_prior_scale': 0.05,
            'yearly_seasonality': False,
            'weekly_seasonality': False,
            'daily_seasonality': False,
            'seasonality_mode': 'additive',
            'seasonality_prior_scale': 2,
            'holidays_prior_scale': 0,
        }

    @staticmethod
    def trial_params():
        params_trial = [
            dict(name='growth', type='categorical', choices=['linear', 'flat']),
            dict(name='n_changepoints', type='int', low=0, high=5),
            dict(name='changepoint_range', type='float', low=0.70, high=1.0),
            dict(name='changepoint_prior_scale', type='float', low=0.001, high=1.0, log=True),
            dict(name='yearly_seasonality', type='categorical', choices=[True, False]),
            dict(name='seasonality_prior_scale', type='float', low=0.001, high=10.0, log=True),
        ]
        return params_trial

    def flexibility(self):
        flexibility = 0

        growth = {'flat': 0.5, 'linear': 1, 'multiplicative': 2}[self.model.growth]
        changepoint_prior_scale = (growth * self.model.changepoint_prior_scale / 0.25) ** 2
        flexibility += changepoint_prior_scale

        flexibility += changepoint_prior_scale * (max(0, (self.model.changepoint_range - 0.80)) / 0.20) ** 2

        seas = self.model.daily_seasonality + self.model.weekly_seasonality + self.model.yearly_seasonality

        if seas > 0:
            seas_mul = 0.5 + 1 * (self.model.seasonality_mode == 'multiplicative')
            flexibility += (seas_mul * self.model.seasonality_prior_scale / 0.25) ** 2

        return flexibility
