from tsfcst.tests.utils import TestModel
from tsfcst.models.inventory import ProphetModel


class TestProphetModel(TestModel):

    def test_general(self):
        self.general(ProphetModel)
