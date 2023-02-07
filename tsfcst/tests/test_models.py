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

    def test_HWModel(self):
        m = HoltWintersSmModel(self.ts_in, params={})
        m.fit()
        f = m.predict(self.n_out)
        assert self.ts_in.data['date'].max() < f.data['date'].min()
        assert len(f) == self.n_out
        assert np.absolute(np.array(f.target) / np.array(self.ts_out.target) - 1).mean() < 0.10

    def test_Prophet(self):
        m = ProphetModel(self.ts_in, params={})
        m.fit()
        f = m.predict(self.n_out)
        assert self.ts_in.data['date'].max() < f.data['date'].min()
        assert len(f) == self.n_out
        assert np.absolute(np.array(f.target) / np.array(self.ts_out.target) - 1).mean() < 0.10
