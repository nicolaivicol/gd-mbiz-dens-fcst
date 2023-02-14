from tsfcst.tests.utils import TestModel
from tsfcst.models.inventory import NaiveModel


class TestHoltWintersSmModel(TestModel):

    def test_general(self):
        self.general(NaiveModel)
