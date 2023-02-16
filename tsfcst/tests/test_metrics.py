import unittest

from tsfcst.utils_tsfcst import smape_cv_opt


class TestMetrics(unittest.TestCase):

    def test_smape_cv_opt(self):
        normal = smape_cv_opt(smape_avg_val=5, smape_std_val=1, smape_avg_fit=3, smape_std_fit=0.5, maprev_val=4, irreg_val=0, irreg_fit=0, irreg_test=0)
        ideal_ = smape_cv_opt(smape_avg_val=3, smape_std_val=0, smape_avg_fit=3, smape_std_fit=0.0, maprev_val=0, irreg_val=0, irreg_fit=0, irreg_test=0)
        lorev_ = smape_cv_opt(smape_avg_val=5, smape_std_val=1, smape_avg_fit=3, smape_std_fit=0.5, maprev_val=1, irreg_val=0, irreg_fit=0, irreg_test=0)
        hirev_ = smape_cv_opt(smape_avg_val=5, smape_std_val=1, smape_avg_fit=3, smape_std_fit=0.5, maprev_val=9, irreg_val=0, irreg_fit=0, irreg_test=0)
        nosd__ = smape_cv_opt(smape_avg_val=5, smape_std_val=0, smape_avg_fit=3, smape_std_fit=0.0, maprev_val=0, irreg_val=0, irreg_fit=0, irreg_test=0)
        hisdfi = smape_cv_opt(smape_avg_val=5, smape_std_val=1, smape_avg_fit=3, smape_std_fit=4.5, maprev_val=4, irreg_val=0, irreg_fit=0, irreg_test=0)
        hisdva = smape_cv_opt(smape_avg_val=5, smape_std_val=5, smape_avg_fit=3, smape_std_fit=0.5, maprev_val=4, irreg_val=0, irreg_fit=0, irreg_test=0)
        diflof = smape_cv_opt(smape_avg_val=5, smape_std_val=1, smape_avg_fit=1, smape_std_fit=0.5, maprev_val=4, irreg_val=0, irreg_fit=0, irreg_test=0)
        difno_ = smape_cv_opt(smape_avg_val=5, smape_std_val=1, smape_avg_fit=5, smape_std_fit=0.5, maprev_val=4, irreg_val=0, irreg_fit=0, irreg_test=0)
        difhif = smape_cv_opt(smape_avg_val=5, smape_std_val=1, smape_avg_fit=9, smape_std_fit=0.5, maprev_val=4, irreg_val=0, irreg_fit=0, irreg_test=0)

        self.assertTrue(abs(3.0 - ideal_) < 0.01)
        self.assertTrue(ideal_ < nosd__)
        self.assertTrue(3.5 < nosd__ < 5.0)
        self.assertTrue(nosd__ < normal)
        self.assertTrue(lorev_ < normal < hirev_)
        self.assertTrue((normal - lorev_) / 3 < (hirev_ - normal) / 5)
        self.assertTrue(normal < hisdfi < hisdva)
        self.assertTrue(diflof < normal < difno_ < difhif)
