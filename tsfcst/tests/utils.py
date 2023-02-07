import pandas as pd


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
