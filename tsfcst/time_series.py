import pandas as pd
import polars as pl
import numpy as np
import json
from typing import Union


class TsData:

    def __init__(
            self,
            dates: Union[pd.Series, np.array, list, str],
            values: Union[pd.Series, np.array, list, str],
            name_date: str = 'date',
            name_value: str = 'value',
            fill_na=0,
            date_format: str = None,
            freq: str = 'MS',  # 'D', 'W', 'M', 'MS', 'A'
    ):
        self.name_date = name_date
        self.name_value = name_value
        self.freq = freq
        self.fill_na = fill_na
        self.data = from_dates_values(
            dates=dates,
            values=values,
            name_date=name_date,
            name_value=name_value,
            fill_na=fill_na,
            date_format=date_format,
            freq=freq,
            as_df=True
        )

    @property
    def time(self) -> pd.Series:
        return self.data[self.name_date]

    @property
    def target(self) -> pd.Series:
        return self.data[self.name_value]

    @staticmethod
    def freq_to_interval_name(freq: str):
        return {'D': 'days', 'W': 'weeks', 'M': 'months', 'MS': 'months'}[freq]

    @property
    def interval_name(self):
        return self.freq_to_interval_name(self.freq)

    def aggregate(self, to_freq) -> 'TsData':
        df = self.data.set_index(self.name_date)
        to_freq_num = self.freq_numeric(to_freq)
        if to_freq == 'W':
            to_freq = 'W-' + max(df.index).strftime('%a')
        df_agg = df.resample(to_freq, label='right').sum(min_count=to_freq_num)
        df_agg = df_agg.loc[~np.isnan(df_agg[self.name_value])]
        ts = TsData(
            dates=df_agg.index,
            values=df_agg[self.name_value],
            name_date=self.name_date,
            name_value=self.name_value,
            freq=to_freq,
            fill_na=self.fill_na,
        )
        return ts

    def disaggregate(self, to_freq='D') -> 'TsData':
        df = self.data.set_index(self.name_date)
        from_freq_num = self.freq_numeric(self.freq)
        new_index = pd.date_range(min(df.index) - pd.DateOffset(from_freq_num-1), max(df.index), freq=to_freq)
        df = df.reindex(new_index).bfill()
        df[self.name_value] = df[self.name_value] / from_freq_num
        ts = TsData(
            dates=df.index,
            values=df[self.name_value],
            name_date=self.name_date,
            name_value=self.name_value,
            freq=to_freq,
            fill_na=self.fill_na,
        )
        return ts

    @staticmethod
    def freq_numeric(freq):
        return {'D': 1, 'W': 7, 'M': 28}[freq[0]]

    def deepcopy(self):
        return TsData(
            dates=self.time,
            values=self.target,
            name_date=self.name_date,
            name_value=self.name_value,
            freq=self.freq,
            fill_na=self.fill_na,
        )

    def __deepcopy__(self):
        return self.deepcopy()

    def __len__(self):
        return len(self.data)

    @staticmethod
    def sample_monthly():
        values = [1249, 1198, 1269, 1243, 1243, 1242, 1217, 1227, 1255, 1257, 1263, 1290, 1328, 1341, 1336,
                  1271, 1256, 1243, 1310, 1326, 1360, 1361, 1359, 1354, 1358, 1344, 1351, 1350, 1386, 1401,
                  1417, 1418, 1433, 1408, 1422, 1461, 1455, 1463, 1472]
        dates = pd.date_range('2019-08-01', periods=len(values), freq='MS')
        return TsData(dates=dates, values=values, freq='MS')


def from_dates_values(dates, values, name_date='date', name_value='value', fill_na=None,
                      date_format=None, origin=None, as_series=False, as_df=False, freq='D') -> pd.DataFrame:

    dates = _convert_to_datetime(dates, date_format, origin)
    assert len(dates) == len(set(dates))

    ts = pd.DataFrame({name_date: dates, name_value: values}).set_index(name_date).sort_index()

    if as_series:
        ts = ts.loc[:, 0]
    elif as_df:
        ts = ts.reset_index()

    return ts


def _convert_to_datetime(dates, date_format=None, origin=None, unit=None):
    if (_is_list_with_type(dates, pd.Timestamp)
            or _is_list_with_type(dates, np.datetime64)
            or (isinstance(dates, pl.Series) and (dates.dtype == pl.Date))
            or isinstance(dates, pd.DatetimeIndex)):
        pass
    elif _is_list_with_type(dates, str) or date_format is not None:
        dates = pd.to_datetime(dates, format=date_format)
    elif _is_list_with_type(dates, int) or _is_list_with_type(dates, float):
        if origin is not None:
            unit = unit if unit is not None else 'D'
            dates = pd.to_datetime(dates, unit=unit, origin=pd.Timestamp(origin))
        else:
            dates = pd.to_datetime(dates, unit='s')  # from unix timestamps
    else:
        raise ValueError(
            '<dates> not valid, it must contain either: \n '
            'a) pd.Timestamp values (no conversion will be performed), \n '
            'b) strings formatted as date (%Y-%m-%d) or datetime (%Y-%m-%d %H:%M:%S or %Y-%m-%dT%H:%M:%S),'
            ' - you can also provide your own date format via <date_format>, '
            '   if <date_format> is provided, the <dates> are treated as strings \n '
            'c) int/float values representing unix timestamps \n '
            'd) int/float values representing time units since the origin date'
            ' - you must provide <origin> (e.g. 2010-01-01) and <unit> (e.g. D, s)'
        )
    return pd.to_datetime(dates)


def _is_list_with_type(x, check_type):
    if isinstance(x, list) or isinstance(x, pd.Series) or isinstance(x, np.ndarray):
        if isinstance(x[0], check_type):
            if all(isinstance(i, check_type) for i in x):
                return True
    return False


def _to_list_if_json_string(x):
    if isinstance(x, str):
        try:
            x = json.loads(x)
            assert isinstance(x, list)
        except:
            raise ValueError("this string is not a valid json list")
    return x


def _is_good_value(x):
    return x is not None and type(x) in [int, float] and not np.isinf(x) and not np.isnan(x)
