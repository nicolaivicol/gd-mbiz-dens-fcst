from tsfcst.tests.utils_tests import TestModel
from tsfcst.models.inventory import ETSsmModel


class TestHoltWintersSmModel(TestModel):

    def test_general(self):
        self.general(ETSsmModel)

    def test_damping(self):
        params_more_damping = {'seasonal': 'no', 'damped_trend': True, 'damping_trend_min': 0.80, 'damping_trend_max': 0.85}
        m_more_damping = ETSsmModel(self.ts_in, params=params_more_damping)
        m_more_damping.fit()
        f_more_damping = m_more_damping.predict(12)

        params_less_damping = {'seasonal': 'no', 'damped_trend': True, 'damping_trend_min': 0.97, 'damping_trend_max': 0.99}
        m_less_damping = ETSsmModel(self.ts_in, params=params_less_damping)
        m_less_damping.fit()
        f_less_damping = m_less_damping.predict(12)

        assert 0.97 <= m_less_damping.model_fit.damping_trend <= 0.99
        assert 0.80 <= m_more_damping.model_fit.damping_trend <= 0.85
        assert all(f_less_damping.target > f_more_damping.target)
        assert m_less_damping.flexibility() > m_more_damping.flexibility()

        try:
            import plotly.graph_objects as go
            import plotly.offline as py
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=self.ts_in.time, y=self.ts_in.target, mode='lines+markers', name='actual'))
            fig.add_trace(go.Scatter(x=f_less_damping.time, y=f_less_damping.target, mode='lines+markers', name='fcst: less damping'))
            fig.add_trace(go.Scatter(x=f_more_damping.time, y=f_more_damping.target, mode='lines+markers', name='fcst: more damping'))
            py.plot(fig)
        except Exception as e:
            pass
