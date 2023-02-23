from tsfcst.tests.utils_tests import TestModel
from tsfcst.models.inventory import ArimaModel


class TestArimaAutoModel(TestModel):

    def test_general(self):
        self.general(ArimaModel)
