import pandas as pd
import numpy as np
import unittest


from tsfcst.time_series import TsData


def is_time_series(ts):
    return ((is_data_frame_with_rows_and_one_column_only(ts)
             or is_series_with_values(ts))
            and has_datetime_index(ts))


def is_data_frame_with_rows_and_cols(df):
    return isinstance(df, pd.DataFrame) and (df.shape[0] > 0) and (df.shape[1] > 0)


def is_data_frame_with_rows_and_one_column_only(df):
    return is_data_frame_with_rows_and_cols(df) & len(df.columns) == 1


def is_series_with_values(s):
    return isinstance(s, pd.Series) and (len(s) > 0)


def has_datetime_index(ts):
    try:
        return isinstance(ts.index, pd.DatetimeIndex)
    except:
        return False


class TestModel(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        n_out = 7
        ts_in = TsData.sample_monthly()
        ts_in.data = ts_in.data.iloc[:-n_out, ]
        ts_out = TsData.sample_monthly()
        ts_out.data = ts_out.data.iloc[-n_out:, ]
        self.n_out, self.ts_in, self.ts_out = n_out, ts_in, ts_out

    def general(self, model_class, th_mape=0.10):
        m = model_class(self.ts_in, params={})
        m.fit()
        f = m.predict(self.n_out)
        assert self.ts_in.data['date'].max() < f.data['date'].min()
        assert len(f) == self.n_out
        assert np.absolute(np.array(f.target) / np.array(self.ts_out.target) - 1).mean() < th_mape
        assert m.flexibility() >= 0

        f_in_sample = m.fitted_values()
        assert all(self.ts_in.time == f_in_sample.time)
        assert len(f_in_sample.target) == len(self.ts_in)
        assert np.absolute(np.array(f_in_sample.target) / np.array(self.ts_in.target) - 1).mean() < th_mape
