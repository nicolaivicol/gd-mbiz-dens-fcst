import numpy as np
import polars as pl
from tsfcst.models.abstract_model import TsModel
import config
from tsfcst.utils_tsfcst import last_n_from_x, rate_diffs


class DriftExogRatesModel(TsModel):
    """
    Drift model using exogeneous rates: last observation with growth rate implied from history
    """
    max_periods = 12  # maximum forecast periods to have rates for
    df_rates_state_country = pl.read_csv(f'{config.DIR_ARTIFACTS}/get_growth_rates/rates-theta.csv')
    df_rates_county = pl.read_csv(f'{config.DIR_ARTIFACTS}/get_rates_submission/rates-best-public-submission.csv')

    def __init__(self, data, params, **kwargs):
        super().__init__(data, params)
        self.last_value = None
        self.rates_county = None
        self.rates_state = None
        self.rates_country = None
        self.rates = None

    @staticmethod
    def propagate_rates(rates, to_size, how='linear'):
        # assume simple rate, to have linear growth in absolute terms (additive model)
        rates = list(rates)
        if len(rates) < to_size:
            r_last = rates[-1] / len(rates)
            if how in ['linear', 'lin', 'add', 'additive']:
                rates_propagation = r_last * np.arange(len(rates) + 1, to_size + 1)
            elif how in ['locf', 'last']:
                rates_propagation = np.repeat(r_last, to_size - len(rates))
            else:
                raise ValueError(f'how={how} not recognized')
            rates = np.concatenate((rates, rates_propagation), axis=None)
        return rates[:to_size]

    def fit(self):
        self.last_value = self.data.target[len(self.data.target)-1]
        max_periods = DriftExogRatesModel.max_periods  # maximum forecast periods to have rates for
        cfips = self.params['cfips']
        asofdate = self.params['asofdate']
        rates_county, rates_state, rates_country = [0.0], [0.0], [0.0]

        df = DriftExogRatesModel.df_rates_county \
            .filter((pl.col('cfips') == int(cfips)) & (pl.col('asofdate') == str(asofdate))) \
            .sort('date')

        if len(df) > 0:
            rates_county = df['rate_county']

        # get state and country rates from dataframe with predicted values
        df = DriftExogRatesModel.df_rates_state_country \
            .filter((pl.col('cfips') == int(cfips)) & (pl.col('asofdate') == str(asofdate))) \
            .sort('date')

        if len(df) > 0:
            rates_state = df['rate_state']
            rates_country = df['rate_country']

        self.rates_county = self.propagate_rates(rates_county, max_periods)
        self.rates_state = self.propagate_rates(rates_state, max_periods)
        self.rates_country = self.propagate_rates(rates_country, max_periods)

        m = self.params['mult_rate']
        w_c = self.params['weight_county']
        w_s = self.params['weight_state']
        w_ct = self.params['weight_country']
        self.rates = m * (w_c * self.rates_county + w_s * self.rates_state + w_ct * self.rates_country)

    def _predict(self, steps):
        assert steps <= DriftExogRatesModel.max_periods
        # 1001, 2023-01-01: 3.3341 = (1 + 0.004481) * 1475 / 44438.0 * 100   ? why 3.3394
        pred = self.last_value * (1 + self.rates[:steps])
        return pred

    def _fitted_values(self):
        return np.array(self.data.target)

    @staticmethod
    def default_params():
        return {
            'cfips': None,  # 1001,
            'asofdate': None,  # '2022-12-01',
            'mult_rate': 1.0,
            'weight_county': 1.0,
            'weight_state': 0.0,
            'weight_country': 0.0,
            'window_drift': 18
        }

    @staticmethod
    def trial_params(trend=None, seasonal=None, multiplicative=None, level=None, damp=None):
        params_trial = [
            dict(name='mult_rate', type='float', low=1.00, high=1.01, step=0.10),
            dict(name='weight_county', type='float', low=1.00, high=1.01, step=0.10),
            dict(name='weight_state', type='float', low=0.0, high=0.01, step=0.10),
            dict(name='weight_country', type='float', low=0.0, high=0.01, step=0.10),
        ]
        return params_trial

    @staticmethod
    def trial_params_full():
        return [
            dict(name='mult_rate', type='float', low=0.0, high=1, step=0.10),
            dict(name='weight_state', type='float', low=0.0, high=1, step=0.10),
            dict(name='weight_country', type='float', low=0.0, high=1, step=0.10),
        ]

    @staticmethod
    def trial_params_grid(trend=True, seasonal=True, multiplicative=True, level=True, damp=False):
        trial_params_ = DriftExogRatesModel.trial_params(trend, seasonal, multiplicative, level, damp)
        return TsModel.trial_params_grid(trial_params_)

    @staticmethod
    def names_params():
        return list(DriftExogRatesModel.default_params().keys())

    def flexibility(self):
        return 0
