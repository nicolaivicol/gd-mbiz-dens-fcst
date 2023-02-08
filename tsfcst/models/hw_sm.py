import numpy as np
from tsfcst.models.abstract_model import TsModel
from statsmodels.tsa.holtwinters import ExponentialSmoothing


class HoltWintersSmModel(TsModel):
    """
    Holt-Winters model with statsmodels implementation

    https://www.statsmodels.org/stable/generated/statsmodels.tsa.holtwinters.ExponentialSmoothing.html
    https://otexts.com/fpp2/holt-winters.html

    Parameters
    trend:
        'add', 'mul', None or any other value for no trend
    damped_trend:
        True, False
    seasonal:
        add', 'mul', None or any other value for no seasonality
    seasonal_periods:
        number of periods in a season
    smoothing_level (alpha):
        is for level smoothing, larger alpha puts more weight on recent levels (observations),
        lower alpha makes the level change smoother (slower)
    smoothing_trend (beta):
        is for trend smoothing, larger beta puts more weight on recent trends, lower beta makes the trend smoother
    smoothing_seasonal (gamma):
        is for seasonality smoothing, larger gamma puts more weight on recent seasons,
        lower gamma makes the seasonality pattern more stable and smoother
    damping_trend (phi):
        is for trend damping, lower phi produces more damping (0: growth is totally damped), larger phi
        produces less damping and the trend can have large growth rates (1: trend not damped at all),
        this parameter is particularly sensitive in the range [0.90, 1.00],
        the sensitivity / effect grows non-linearly as it approaches 1.00
    """

    def __init__(self, data, params, **kwargs):
        super().__init__(data, params)
        self.set_params()
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
            initial_trend=initial_trend if self.params['trend'] is not None else None,
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

    def set_params(self):
        if self.params['trend'] not in ['add', 'mul']:
            self.params['trend'] = None
            self.params['damped_trend'] = False
            self.params['smoothing_trend_min'] = 0
            self.params['smoothing_trend_max'] = 0

        if self.params['seasonal'] not in ['add', 'mul']:
            self.params['seasonal'] = None
            self.params['smoothing_seasonal_min'] = 0
            self.params['smoothing_seasonal_max'] = 0

        if not self.params['damped_trend']:
            self.params['damping_trend_min'] = 1
            self.params['damping_trend_max'] = 1

    @staticmethod
    def default_params():
        return {
            'trend': 'add',
            'damped_trend': True,
            'seasonal': 'add',
            'seasonal_periods': 12,
            'smoothing_level_min': 0,
            'smoothing_level_max': 0.05,
            'smoothing_trend_min': 0,
            'smoothing_trend_max': 0.04,
            'smoothing_seasonal_min': 0,
            'smoothing_seasonal_max': 0.05,
            'damping_trend_min': 0.90,
            'damping_trend_max': 0.98,
        }

    @staticmethod
    def trial_params():
        params_trial = [
            dict(name='trend', type='categorical', choices=['add', 'no']),
            dict(name='seasonality', type='categorical', choices=['add', 'no']),
            dict(name='damped_trend', type='categorical', choices=[True, False]),
            dict(name='smoothing_level_max', type='float', low=0.0001, high=0.33, log=True),
            dict(name='smoothing_trend_max', type='float', low=0.0001, high=0.33, log=True),
            dict(name='smoothing_seasonal_max', type='float', low=0.0001, high=0.33, log=True),
            dict(name='damping_trend_min', type='float', low=0.70, high=0.94),
            dict(name='damping_trend_max', type='float', low=0.95, high=0.995),
        ]
        return params_trial

    def flexibility(self):
        flexibility = 0
        flexibility += self.model_fit.params['smoothing_level'] ** 2

        if self.model_fit.model.has_trend:
            if self.model_fit.model.damped_trend is False:
                not_damped_trend = 1
            else:
                not_damped_trend = self.model_fit.params['damping_trend'] ** 2

            flexibility += not_damped_trend

            if self.model_fit.model.trend == 'add':
                flexibility += (1 + not_damped_trend) * self.model_fit.params['smoothing_trend'] ** 2
            else:
                flexibility += (2 + not_damped_trend) * self.model_fit.params['smoothing_trend'] ** 2

        if self.model_fit.model.has_seasonal:
            if self.model_fit.model.seasonal == 'add':
                flexibility += self.model_fit.params['smoothing_seasonal'] ** 2
            else:
                flexibility += 2 * self.model_fit.params['smoothing_seasonal'] ** 2

        return flexibility

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
