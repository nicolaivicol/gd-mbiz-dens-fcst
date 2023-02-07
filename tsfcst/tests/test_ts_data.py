import unittest
import pandas as pd

from tsfcst.time_series import TsData


class TestTsData(unittest.TestCase):

    def test_default(self):
        ts = TsData(dates=['2021-01-01', '2021-02-01'], values=[1, 2], freq='MS')
        assert len(ts) == 2
        assert all(ts.time == pd.to_datetime(['2021-01-01', '2021-02-01']))
        assert all(ts.target == [1, 2])
