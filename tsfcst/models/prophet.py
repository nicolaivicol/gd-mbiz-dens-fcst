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
        "linear" or "logistic".
        "Linear" means the trend will grow at a constant rate
        "logistic" means the growth rate will start high and then taper off over time.
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
            'n_changepoints': 4,
            'changepoint_range': 0.95,
            'changepoint_prior_scale': 0.05,
            'yearly_seasonality': 'auto',
            'weekly_seasonality': False,
            'daily_seasonality': False,
            'seasonality_mode': 'additive',
            'seasonality_prior_scale': 5,
            'holidays_prior_scale': 0,
        }

    @staticmethod
    def trial_params(trial):
        params_trial = {
            'growth': trial.suggest_categorical('growth', ['linear', 'flat']),
            'n_changepoints': trial.suggest_int('n_changepoints', 0, 5),
            'changepoint_range': trial.suggest_float('changepoint_range', 0.70, 1.0),
            'changepoint_prior_scale': trial.suggest_float('changepoint_prior_scale', 0.001, 0.99, log=True),
            'yearly_seasonality': trial.suggest_categorical('yearly_seasonality', [True, False]),
            'seasonality_prior_scale': trial.suggest_float('seasonality_prior_scale', 0.001, 10, log=True),
        }
        return params_trial
