from tsfcst.tests.utils_tests import TestModel
from tsfcst.models.inventory import ThetaSmModel


class TestThetaSmModel(TestModel):

    def test_general(self):
        self.general(ThetaSmModel)

    def test_thetas(self):
        fcsts = {}

        for theta in [1, 1.5, 2, 5, 10]:
            m = ThetaSmModel(self.ts_in, params={'theta': theta})
            m.fit()
            f = m.predict(12)
            fcsts[f'theta={theta}'] = f

        assert all(fcsts['theta=1'].target < fcsts['theta=10'].target)

        try:
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

        except Exception as e:
            pass
