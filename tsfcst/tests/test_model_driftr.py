from tsfcst.tests.utils_tests import TestModel
from tsfcst.models.inventory import DriftExogRatesModel


class TestDriftExogRatesModel(TestModel):

    def test_general(self):
        self.general(DriftExogRatesModel, params={'cfips': 1001, 'asofdate': '2022-07-01'})
