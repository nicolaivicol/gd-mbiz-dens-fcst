import unittest

from tsfcst.time_series import TsData
from tsfcst.utils_tsfcst import get_feats, mean_prc_change, prc_change_lte, smape_wrt_avg


class TestTsFeats(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.ts = TsData.sample_monthly()

    def test_all(self):
        feats = get_feats(self.ts.target)
        self.assertTrue(all([isinstance(f, float) for f in feats.values()]))
        print('\n'.join([f'{k}={v}' for k, v in feats.items()]))

    def test_prc_change_lte(self):
        self.assertEqual(0.75, prc_change_lte([100, 100, 101, 101, 101], small=0))
        self.assertEqual(1.00, prc_change_lte([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], small=0))
        self.assertEqual(1.00, prc_change_lte([1, 1, 1, 0, 0, 0, 0, 0, 0, 0], small=0, last_n=5))

    def test_smape_wrt_avg(self):
        self.assertAlmostEqual(1.0, smape_wrt_avg([99, 101, 99, 101]), 3)
        self.assertAlmostEqual(1.0, smape_wrt_avg([99, 101, 99, 101], last_n=2), 3)
        self.assertAlmostEqual(0.0, smape_wrt_avg([100, 100, 100, 100]), 3)
        self.assertAlmostEqual(0.0, smape_wrt_avg([0, 0, 0, 0]), 3)
