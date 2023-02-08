import unittest
import numpy as np

from tsfcst.time_series import TsData
from tsfcst.models.inventory import MovingAverageModel, HoltWintersSmModel, ProphetModel


class TestModels(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        n_out = 7
        ts_in = TsData.sample_monthly()
        ts_in.data = ts_in.data.iloc[:-n_out, ]
        ts_out = TsData.sample_monthly()
        ts_out.data = ts_out.data.iloc[-n_out:, ]
        self.n_out, self.ts_in, self.ts_out = n_out, ts_in, ts_out

    def test_MovingAverageModel(self):
        m = MovingAverageModel(self.ts_in, params={'average': 'exponential', 'window': 24})
        m.fit()
        f = m.predict(self.n_out)
        assert self.ts_in.data['date'].max() < f.data['date'].min()
        assert len(f) == self.n_out
        assert np.absolute(np.array(f.target) / np.array(self.ts_out.target) - 1).mean() < 0.10
        assert m.flexibility() > 0

    def test_HWModel(self):
        m = HoltWintersSmModel(self.ts_in, params={})
        m.fit()
        f = m.predict(self.n_out)
        assert self.ts_in.data['date'].max() < f.data['date'].min()
        assert len(f) == self.n_out
        assert np.absolute(np.array(f.target) / np.array(self.ts_out.target) - 1).mean() < 0.10
        assert m.flexibility() > 0

    def test_HWModel_damping(self):
        plot = False

        params_hi_damping = {'seasonal': 'no', 'damped_trend': True, 'damping_trend_min': 0.97, 'damping_trend_max': 0.99}
        m_hi_damping = HoltWintersSmModel(self.ts_in, params=params_hi_damping)
        m_hi_damping.fit()
        f_hi_damping = m_hi_damping.predict(12)

        params_lo_damping = {'seasonal': 'no', 'damped_trend': True, 'damping_trend_min': 0.80, 'damping_trend_max': 0.85}
        m_lo_damping = HoltWintersSmModel(self.ts_in, params=params_lo_damping)
        m_lo_damping.fit()
        f_lo_damping = m_lo_damping.predict(12)

        assert 0.97 <= m_hi_damping.model_fit.params['damping_trend'] <= 0.99
        assert 0.80 <= m_lo_damping.model_fit.params['damping_trend'] <= 0.85
        assert all(f_hi_damping.target > f_lo_damping.target)
        assert m_lo_damping.flexibility() > m_hi_damping.flexibility()

        if plot:
            import plotly.graph_objects as go
            import plotly.offline as py

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=self.ts_in.time, y=self.ts_in.target, mode='lines+markers', name='actual'))
            fig.add_trace(go.Scatter(x=f_lo_damping.time, y=f_lo_damping.target, mode='lines+markers', name='fcst: lo_damping'))
            fig.add_trace(go.Scatter(x=f_hi_damping.time, y=f_hi_damping.target, mode='lines+markers', name='fcst: hi_damping'))
            py.plot(fig)

    def test_Prophet(self):
        m = ProphetModel(self.ts_in, params={})
        m.fit()
        f = m.predict(self.n_out)
        assert self.ts_in.data['date'].max() < f.data['date'].min()
        assert len(f) == self.n_out
        assert np.absolute(np.array(f.target) / np.array(self.ts_out.target) - 1).mean() < 0.10
        assert m.flexibility() > 0
