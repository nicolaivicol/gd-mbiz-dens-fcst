from tsfcst.tests.utils_tests import TestModel
from tsfcst.models.inventory import DriftModel


class TestDriftModel(TestModel):

    def test_general(self):
        self.general(DriftModel)
