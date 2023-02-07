import numpy as np
from tsfcst.models.abstract_model import TsModel
from statsmodels.tsa.holtwinters import ExponentialSmoothing


class HoltWintersSmModel(TsModel):
    """
    Holt-Winters model with statsmodels implementation
    https://www.statsmodels.org/stable/generated/statsmodels.tsa.holtwinters.ExponentialSmoothing.html
    https://otexts.com/fpp2/holt-winters.html

    Parameters
    smoothing_level (alpha):
        The smoothing factor for the level component.
        A higher alpha places more weight on recent observations and a lower
        alpha places more weight on historical observations.
    smoothing_trend (beta):
        The smoothing factor for the trend component.
        A higher beta will result in a stronger reaction to changes in the
        trend, while a lower beta will result in a smoother trend estimate.
    smoothing_seasonal (gamma):
        The smoothing factor for the seasonal component.
        A higher gamma will result in a stronger reaction to changes in the
        seasonal component, while a lower gamma will result in a smoother seasonal estimate.
    damping_trend (phi):
        The damping factor, used to control the magnitude of the seasonal fluctuations.
        A value of 1 means that the seasonal fluctuations are fully damped,
        and a value of 0 means no damping.
    """

    def __init__(self, data, params, **kwargs):
        super().__init__(data, params)
        self.model = None
        self.model_fit = None

    def fit(self):
        initialization_method, initial_level, initial_trend, initial_seasonal = self._initial_params()
        self.model = ExponentialSmoothing(
            endog=self.data.target,
            trend=self.params['trend'],
            damped_trend=self.params['damped_trend'],
            seasonal=self.params['seasonal'],
            seasonal_periods=self.params['seasonal_periods'] if self.params['seasonal'] is not None else None,
            initialization_method=initialization_method,
            dates=self.data.time,
            freq=self.data.freq,
            initial_trend=initial_trend,
            initial_level=initial_level,
            initial_seasonal=initial_seasonal if self.params['seasonal'] is not None else None,
            bounds={
                'smoothing_level': (self.params['smoothing_level_min'], self.params['smoothing_level_max']),
                'smoothing_trend': (self.params['smoothing_trend_min'], self.params['smoothing_trend_max']),
                'smoothing_seasonal': (self.params['smoothing_seasonal_min'], self.params['smoothing_seasonal_max']),
                'damping_trend': (self.params['damping_trend_min'], self.params['damping_trend_max']),
            },
        )
        self.model_fit = self.model.fit()

    def _predict(self, steps):
        return self.model_fit.forecast(steps)

    def _fitted_values(self):
        return np.array(self.model_fit.fittedvalues)

    @staticmethod
    def default_params():
        return {
            'trend': 'add',  # 'add' or 'mul'
            'damped_trend': True,  # True or False
            'seasonal': 'add',  # 'add' or 'mul'
            'seasonal_periods': 12,  # number of periods in a season
            # alpha: larger values of alpha indicate more weight on recent observations
            'smoothing_level_min': 0,
            'smoothing_level_max': 0.05,
            # beta: larger values of beta indicate more weight on recent trends
            'smoothing_trend_min': 0,
            'smoothing_trend_max': 0.04,
            # gamma: larger values of gamma indicate more weight on recent seasons
            'smoothing_seasonal_min': 0,
            'smoothing_seasonal_max': 0.05,
            # phi: larger values of phi indicate more damping
            'damping_trend_min': 0.90,
            'damping_trend_max': 0.98,
        }

    @staticmethod
    def trial_params(trial):
        params_trial = {
            # 'trend': trial.suggest_categorical('trend', ['add', 'mul']),
            'damped_trend': trial.suggest_categorical('damped_trend', [True, False]),
            'smoothing_level_max': trial.suggest_float('smoothing_level_max', 0.0001, 0.33, log=True),
            'smoothing_trend_max': trial.suggest_float('smoothing_trend_max', 0.0001, 0.33, log=True),
            'smoothing_seasonal_max': trial.suggest_float('smoothing_seasonal_max', 0.0001, 0.33, log=True),
            'damping_trend_min': trial.suggest_float('damping_trend_min', 0.70, 0.91),
            'damping_trend_max': trial.suggest_float('damping_trend_max', 0.95, 0.995),
        }
        return params_trial

    def _initial_params(self):
        """ handle the case when there are not enough observations for the 'heuristic' initialization """
        initialization_method, initial_level, initial_trend, initial_seasonal = 'heuristic', None, None, None
        seasonal_periods = self.params['seasonal_periods'] if self.params['seasonal'] is not None else 0
        if len(self.data.target) < (2 * seasonal_periods + 10):
            n_seas = self.params['seasonal_periods']
            initialization_method = 'known'
            initial_level = np.nanmedian(self.data.target[:min(len(self.data.target), max(n_seas, 5))])
            initial_trend = 0 if self.params['trend'] == 'add' else 1
            initial_seasonal = np.repeat(0, n_seas) if self.params['trend'] == 'add' else np.repeat(1, n_seas)
        return initialization_method, initial_level, initial_trend, initial_seasonal
