import unittest
import numpy as np

from tsfcst.time_series import TsData
from tsfcst.models.inventory import ThetaSmModel


class TestThetaSmModel(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        n_out = 7
        ts_in = TsData.sample_monthly()
        ts_in.data = ts_in.data.iloc[:-n_out, ]
        ts_out = TsData.sample_monthly()
        ts_out.data = ts_out.data.iloc[-n_out:, ]
        self.n_out, self.ts_in, self.ts_out = n_out, ts_in, ts_out

    def test_ThetaSmModel(self):
        m = ThetaSmModel(self.ts_in, params={'theta': 1})
        m.fit()
        f = m.predict(self.n_out)
        assert self.ts_in.data['date'].max() < f.data['date'].min()
        assert len(f) == self.n_out
        assert np.absolute(np.array(f.target) / np.array(self.ts_out.target) - 1).mean() < 0.10
        assert m.flexibility() > 0

    def test_ThetaSmModel_thetas(self):
        plot = True
        fcsts = {}

        for theta in [1, 1.5, 2, 5, 10]:
            m = ThetaSmModel(self.ts_in, params={'theta': theta})
            m.fit()
            f = m.predict(12)
            fcsts[f'theta={theta}'] = f

        if plot:
            import plotly.graph_objects as go
            import plotly.offline as py

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=self.ts_in.time, y=self.ts_in.target, mode='lines+markers', name='actual'))

            for i, name_fcst in enumerate(fcsts.items()):
                name, f = name_fcst
                fig.add_trace(
                    go.Scatter(
                        x=f.time,
                        y=f.target,
                        mode='lines+markers',
                        name=name,
                        opacity=0.4 + i / len(fcsts) * 0.50,
                        line=dict(color='red')
                    )
                )
            py.plot(fig)

