import unittest
import numpy as np
import pandas as pd

from tsfcst.models.inventory import NaiveModel, MovingAverageModel, ThetaSmModel, HoltWintersSmModel
from tsfcst.forecasters.forecaster import ForecasterConfig
from tsfcst.time_series import TsData
from tsfcst.ensemble import Ensemble


class TestEnsemble(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.ts = TsData.sample_monthly()
        self.cfgs = {
            'naive': ForecasterConfig(NaiveModel, {}, {}),
            'ma': ForecasterConfig(MovingAverageModel, {}, {}),
            'theta': ForecasterConfig(ThetaSmModel, {}, {}),
            'hw': ForecasterConfig(HoltWintersSmModel, {}, {}),
        }

    def test_forecast(self):
        ens = Ensemble(data=self.ts, fcster_configs=self.cfgs)
        fcst = ens.forecast(7)
        self.assertEqual(['date', 'naive', 'ma', 'theta', 'hw', 'ensemble'], list(fcst.columns))
        self.assertEqual(7, len(fcst))
        avg_models = np.array(fcst.select(list(self.cfgs.keys())).mean(axis=1))
        self.assertTrue(all(np.abs(avg_models - np.array(fcst['ensemble'])) < 0.01))

    def test_forecast_median(self):
        ens = Ensemble(data=self.ts, fcster_configs=self.cfgs, method='median')
        fcst = ens.forecast(7)
        self.assertEqual(['date', 'naive', 'ma', 'theta', 'hw', 'ensemble'], list(fcst.columns))
        self.assertEqual(7, len(fcst))
        avg_models = np.array(fcst.select(list(self.cfgs.keys())).mean(axis=1))
        self.assertTrue(all(np.abs(avg_models - np.array(fcst['ensemble'])) < 0.01))
