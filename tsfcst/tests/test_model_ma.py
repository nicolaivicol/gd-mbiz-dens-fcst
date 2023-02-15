from tsfcst.tests.utils_tests import TestModel
from tsfcst.models.inventory import MovingAverageModel


class TestMovingAverageModel(TestModel):

    def test_general(self):
        self.general(MovingAverageModel)
