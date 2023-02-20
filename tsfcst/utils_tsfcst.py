import pandas as pd
import numpy as np
import plotly.graph_objects as go
import collections.abc
from statsmodels.tsa.seasonal import STL
import math
import statsmodels.api as sm
from scipy import stats
from itertools import groupby
import polars as pl
from tsfcst.chow_test import chow_test


def smape(y_true, y_pred, weight=None):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    nominator = np.abs(y_true - y_pred)
    error_prc = nominator / denominator * 100.0
    # if both y_true and y_pred are zeros, then error_prc is 0
    error_prc[denominator == 0] = 0.0

    # # ignore nan values
    # if ignore_na:
    #     idx = ~np.isnan(error_prc)
    #     error_prc = error_prc[idx]
    #     if weight is not None:
    #         weight = np.array(weight)[idx]
    #
    # if len(error_prc) == 0:
    #     return np.NaN

    # mean absolute percentage error
    if weight is None:
        r = np.nanmean(error_prc)
    else:
        r = np.nansum(error_prc * weight) / np.nansum(weight)

    return r


def smape_cv_opt(smape_avg_val, smape_std_val, smape_avg_fit, smape_std_fit, maprev_val, irreg_val, irreg_fit,
                 irreg_test=0, **kwargs):
    smape_weighted_fit_val = 0.8 * smape_avg_val + 0.2 * smape_avg_fit
    std_weighted_fit_val = 0.8 * smape_std_val + 0.2 * smape_std_fit
    penalty_std = max(0.50 * std_weighted_fit_val, 0.2 * std_weighted_fit_val ** 2)
    diff_fit_val = smape_avg_fit - smape_avg_val
    penalty_diff = 0.10 * max(0, -diff_fit_val) + 0.10 * max(0, diff_fit_val) ** 2
    penalty_rev = 0.05 * maprev_val ** 2
    penalty_irreg = 0.05 * (0.6 * irreg_val + 0.2 * irreg_fit + 0.2 * irreg_test)
    return smape_weighted_fit_val + penalty_std + penalty_diff + penalty_rev + penalty_irreg


def irreg_rate(x):
    x = np.array(x)
    if len(x) <= 2:
        return 0

    diffs = x[1:] - x[:-1]
    denominator = ((np.abs(x[:-1]) + np.abs(x[1:]) + 0.0001) / 2)
    prc_diffs = diffs / denominator
    squared_prc_diffs = np.square(prc_diffs)

    if all(np.isnan(squared_prc_diffs)):
        return 0

    squared_prc_diffs = np.nanmean(squared_prc_diffs)

    return np.sqrt(squared_prc_diffs)*100


def calc_fcst_error_metrics(df_ts, df_fcsts, time_name='date', target_name='value'):

    if isinstance(df_ts, pd.DataFrame):
        df_ts = pl.DataFrame(df_ts).with_columns(pl.col(time_name).cast(pl.Date))

    if isinstance(df_fcsts, pd.DataFrame):
        df_fcsts = pl.DataFrame(df_fcsts).with_columns(pl.col(time_name).cast(pl.Date))

    df = df_ts[[time_name, target_name]].join(df_fcsts, on=time_name).sort(time_name)

    cols_fcst = [c for c in df_fcsts.columns if c != time_name]

    smape_ = []
    irreg_ = []
    for c in cols_fcst:
        smape_.append(smape(np.array(df[target_name]), np.array(df[c])))
        irreg_.append(irreg_rate(np.array(df[c])))

    # symmetric mean absolute percentage error, avg and std
    smape_avg = round(np.nanmean(smape_), 4)
    smape_std = round(np.nanstd(smape_), 4)

    # mean absolute percentage revision of forecasts
    fcst_avg = df[cols_fcst].mean(axis=1).to_numpy()
    fcst_min = df[cols_fcst].min(axis=1).to_numpy()
    fcst_max = df[cols_fcst].max(axis=1).to_numpy()

    if any(fcst_avg == 0):
        maprev = round(np.nanmean((fcst_max - fcst_min) / np.nanmean(np.abs(fcst_avg)+0.00001)) * 100, 4)
    else:
        maprev = round(np.nanmean((fcst_max - fcst_min)/np.abs(fcst_avg)) * 100, 4)

    irreg = round(np.nanmean(irreg_), 4)

    return {'smape_avg': smape_avg, 'smape_std': smape_std, 'maprev': maprev, 'irreg': irreg}


def plotly_add_time_slider(fig, visible=True):
    fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=3, label="3m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(visible=visible),
            type="date"
        )
    )
    return fig


def plot_fcsts_and_actual(df_actual, df_fcsts, time_name='date', target_name='value', freq='MS', colors=None):
    df_idx = pd.DataFrame({time_name: pd.date_range(np.min(df_actual[time_name]), np.max(df_actual[time_name]), freq=freq)})
    df_actual = pd.merge(df_idx, df_actual, how='outer', on=[time_name]).fillna(0)
    df = pd.merge(df_fcsts, df_actual, how='outer', on=[time_name]).sort_values(by=[time_name])
    cols_fcst = [c for c in df_fcsts.columns if c != time_name]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df[time_name],
            y=np.round(df[target_name], 4),
            name=f'actual',
            mode='lines+markers',
            opacity=0.7,
            line=dict(color='black', width=2))
    )
    for i, c in enumerate(sorted(cols_fcst)):
        tmp = df.loc[df[c].notnull()] #  & df[c] >= 0

        if colors is not None:
            opacity_ = 0.6
            color_ = colors.get(c, 'red') if isinstance(colors, dict) else colors[min(i, len(colors) - 1)]
            line_ = dict(color=color_, width=2)
            if c == 'ensemble':
                opacity_ = 0.8
                line_ = dict(color=color_, width=3)
        else:
            opacity_ = 0.4 + i * 0.4 / len(cols_fcst)
            line_ = dict(color='red', width=2)

        fig.add_trace(
            go.Scatter(
                x=tmp[time_name],
                y=np.round(tmp[c], 4),
                name=c,
                mode='lines+markers',
                opacity=opacity_,
                line=line_
            )
        )
    min_ = np.min(df[[target_name] + cols_fcst].min(axis=1, skipna=True))
    max_ = np.max(df[[target_name] + cols_fcst].max(axis=1, skipna=True))
    fig.update_layout(autosize=True, legend=dict(x=0, y=1, bgcolor='rgba(0,0,0,0)'),
                      yaxis_range=[min(max(0.95*min_, 0.90*max_), 0.98*min_),
                                   max(min(1.05*max_, 1.10*min_), 1.02*max_)],
                      height=650, width=1200, margin=dict(l=25, r=25, t=25, b=25))
    # py.plot(fig)
    return fig


def update_nested_dict(d, u):
    if u is None:
        return d
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update_nested_dict(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def chowtest(y, last_index):
    chow_value, p_value = chow_test(
        X_series=pd.Series(range(len(y))),
        y_series=pd.Series(y),
        last_index=int(last_index),
        first_index=int(last_index) + 1,
    )
    return chow_value, p_value


def last_n_from_x(x, last_n=None) -> np.ndarray:
    x = np.array(x)
    if last_n is not None:
        x = x[-min(len(x), last_n):]
    return x


def prc_diffs(x):
    prc_diffs = np.diff(x) / np.maximum(x[:-1], 0.001)
    prc_diffs = np.maximum(np.minimum(0.9999, prc_diffs), -0.9999)
    return prc_diffs * 100


def prc_zeros(x, last_n=None):
    x = last_n_from_x(x, last_n)
    return np.mean(np.array(x) == 0)


def prc_lte(x, small, last_n=None):
    x = last_n_from_x(x, last_n)
    return np.mean(np.array(x) <= small)


def prc_gt(x, threshold, last_n=None):
    x = last_n_from_x(x, last_n)
    return np.mean(np.array(x) > threshold)


def sd_m_ratio(x, last_n=None):
    x = last_n_from_x(x, last_n)
    return min(9.99, np.nanstd(np.array(x))/(np.nanmean(np.abs(np.array(x))) + 0.001))


def prc_change_lte(x, small, last_n=None):
    x = last_n_from_x(x, last_n)
    return np.mean(np.abs(prc_diffs(x)) <= small)


def mean_prc_change(x, last_n=None):
    x = last_n_from_x(x, last_n)
    return np.mean(prc_diffs(x))


def smape_wrt_avg(x, last_n=None):
    x = last_n_from_x(x, last_n)
    avg = np.nanmean(x)
    denom = (np.abs(avg) + np.abs(x)) / 2
    abs_prc_errors = np.abs(x - avg) / denom
    abs_prc_errors[denom == 0] = 0
    mean_abs_prc_errors = np.nanmean(abs_prc_errors) * 100
    mean_abs_prc_errors = min(99.999, mean_abs_prc_errors)
    return mean_abs_prc_errors


def avg(x, last_n=None):
    x = last_n_from_x(x, last_n)
    return np.nanmean(x)


def perc(x, p=50, last_n=None):
    x = last_n_from_x(x, last_n)
    return np.nanpercentile(x, p)


def iqr_m_ratio(x, q_up=75, q_lo=25, last_n=None):
    x = last_n_from_x(x, last_n)
    q_up_ = np.nanpercentile(x, q_up)
    q_lo_ = np.nanpercentile(x, q_lo)
    return min(99, (q_up_ - q_lo_) / (q_up_ + 0.0001))


def n_gt(x, small, last_n=None):
    if last_n is None:
        return np.nansum(np.array(x) > small)
    else:
        x_last = x[-min(len(x), last_n):]
        return np.nansum(np.array(x_last) > small)


def q_trend(x, last_n):
    return (perc(x, 50, last_n) - perc(x, 10))/(perc(x, 90) - perc(x, 10) + 0.001)


def get_lin_reg_summary(y, last_n=None):
    y = last_n_from_x(y, last_n)
    try:
        slope, intercept, r, p, se = stats.linregress(np.arange(len(y)), y)
        avg_abs_y = np.mean(np.abs(y))
        try:
            slope = slope / avg_abs_y
        except ZeroDivisionError:
            slope = 0
    except:
        slope, intercept, r, p, se = 0, 0, 0, 1, 99

    return {'slope': slope, 'intercept': intercept, 'r_squared': r**2, 'p_value': p, 'se': se}


def get_stability(x: np.ndarray, window_size: int = 6, last_n=None) -> float:
    x = last_n_from_x(x, last_n)
    means_ = np.array([np.mean(x_w) for x_w in np.array_split(x, len(x) // window_size + 1)])
    prc_abs_diffs = np.abs(means_ / np.mean(x) - 1)
    return np.mean(prc_abs_diffs) * 100


def get_lumpiness(x: np.ndarray, window_size: int = 10) -> float:
    v = [np.var(x_w) for x_w in np.array_split(x, len(x) // window_size + 1)]
    return np.nanvar(v)


def get_crossing_points(x: np.ndarray, last_n=None) -> float:
    x = last_n_from_x(x, last_n)
    median_ = np.nanmedian(x)
    cp = 0
    for i in range(len(x) - 1):
        if x[i] <= median_ < x[i + 1] or x[i] >= median_ > x[i + 1]:
            cp += 1
    return cp


def get_binarize_mean(x: np.ndarray, last_n=None) -> float:
    x = last_n_from_x(x, last_n)
    return np.nanmean(x > np.nanmean(x))


def get_flat_spots(x: np.ndarray, nbins: int = 5) -> int:
    x = np.array(x)

    if len(x) <= nbins:
        return np.nan

    max_run_length = 0
    window_size = int(len(x) / nbins)
    for i in range(0, len(x), window_size):
        run_length = np.max([len(list(v)) for k, v in groupby(x[i: i + window_size])])
        if run_length > max_run_length:
            max_run_length = run_length
    return max_run_length


def get_hurst(x, lag_size: int = 12) -> float:
    """
    Getting: Hurst Exponent wiki: https://en.wikipedia.org/wiki/Hurst_exponent
    Args:
        x: The univariate time series array in the form of 1d numpy array.
        lag_size: int; Size for getting lagged time series data.
    Returns:
        The Hurst Exponent of the time series array
    """

    x = np.array(x)

    # Create the range of lag values
    lags = range(2, min(lag_size, len(x) - 1))

    # Calculate the array of the variances of the lagged differences
    tau = [np.std(np.asarray(x)[lag:] - np.asarray(x)[:-lag]) for lag in lags]

    # Use a linear fit to estimate the Hurst Exponent
    poly = np.polyfit(np.log(lags), np.log(tau), 1)

    # Return the Hurst exponent from the polyfit output
    return poly[0] if not np.isnan(poly[0]) else 0


def trunc_num_values_in_dict_to_min_max(dict_, min_val=-99999, max_val=99999):
    for k, v in dict_.items():
        try:
            dict_[k] = max(min_val, min(max_val, np.round(v, 3)))
        except:
            pass
    return dict_


def get_feats(x, min_val=-9999, max_val=9999):
    x = np.array(x)
    lin_reg_summary_ = get_lin_reg_summary(x)
    lin_reg_summary_10 = get_lin_reg_summary(x, 10)
    feats = {
        'avg': avg(x),
        'smape2avg': smape_wrt_avg(x),
        'smape2avg_3': smape_wrt_avg(x, 3),
        'smape2avg_5': smape_wrt_avg(x, 5),
        'smape2avg_10': smape_wrt_avg(x, 10),
        'smape2avg_20': smape_wrt_avg(x, 20),
        'iqr_m_ratio': iqr_m_ratio(x),
        'iqr_m_ratio_5': iqr_m_ratio(x, last_n=5),
        'iqr_m_ratio_10': iqr_m_ratio(x, last_n=10),
        'prc_zeros': prc_zeros(x),
        'prc_zeros_1': prc_zeros(x, 1),
        'prc_zeros_5': prc_zeros(x, 5),
        'prc_small': prc_lte(x, small=20),
        'prc_small_10': prc_lte(x, small=20, last_n=10),
        'prc_not_small': prc_gt(x, threshold=100),
        'prc_not_small_10': prc_gt(x, threshold=100, last_n=10),
        'prc_change_lte_0': prc_change_lte(x, small=0),
        'prc_change_lte_0_10': prc_change_lte(x, small=0, last_n=10),
        'prc_change_lte_1': prc_change_lte(x, small=1),
        'prc_change_lte_1_10': prc_change_lte(x, small=1, last_n=10),
        'q_trend_1': q_trend(x, 1),
        'q_trend_5': q_trend(x, 5),
        'q_trend_10': q_trend(x, 10),
        'r_sq': lin_reg_summary_['r_squared'],
        'slope': lin_reg_summary_['slope'],
        'p_value': lin_reg_summary_['p_value'],
        'r_sq_10': lin_reg_summary_10['r_squared'],
        'slope_10': lin_reg_summary_10['slope'],
        'p_value_10': lin_reg_summary_10['p_value'],
        'stability': get_stability(x),
        # 'lumpiness': get_lumpiness(x),
        # 'n_cross_pts': get_crossing_points(x),
        'prc_cross_pts': get_crossing_points(x) / len(x),
        'prc_cross_pts_10': get_crossing_points(x, 10) / min(len(x), 10),
        'bin_mean': get_binarize_mean(x),
        'bin_mean_10': get_binarize_mean(x, 10),
        'prc_flat_spots': get_flat_spots(x) / len(x),
        'hurst_5': get_hurst(x, 5),
        'hurst_10': get_hurst(x, 10),
    }
    feats = trunc_num_values_in_dict_to_min_max(feats, min_val, max_val)
    return feats


def get_stats(values, by):
    d = pd.DataFrame({'values': np.array(values).astype('float'), 'by': np.array(by)})
    d_agg = d.groupby('by').agg(
        count=('values', len),
        min=('values', np.nanmin),
        p10=('values', lambda x: np.nanpercentile(x, 10)),
        p25=('values', lambda x: np.nanpercentile(x, 25)),
        p50=('values', lambda x: np.nanpercentile(x, 50)),
        mean=('values', np.nanmean),
        sd=('values', np.nanstd),
        p75=('values', lambda x: np.nanpercentile(x, 75)),
        p90=('values', lambda x: np.nanpercentile(x, 90)),
        max=('values', np.nanmax)
    ).reset_index()
    return d_agg


def get_box_chart(x, y, name=None, **kwargs):
    d_agg = get_stats(y, x)
    box = go.Box(
        name=name,
        x=d_agg['by'],
        lowerfence=d_agg['p10'],
        q1=d_agg['p25'],
        median=d_agg['p50'],
        mean=d_agg['mean'],
        q3=d_agg['p75'],
        upperfence=d_agg['p90'],
        # boxpoints=False,
        # boxmean=True,
        **kwargs
    )
    return box


def std_trim(y, sd_prc_lo=1, sd_prc_hi=99):
    q_lo, q_hi = np.nanpercentile(y, sd_prc_lo), np.nanpercentile(y, sd_prc_hi)
    is_btw_hi_lo = (q_lo < y) & (y < q_hi)
    if (len(y) - np.nansum(is_btw_hi_lo)) > 2:
        y = y[is_btw_hi_lo]
    std = np.nanstd(y)
    return std


def nextodd(x):
    return int(x / 2) * 2 + 1


class STLparams():
    _FREQUENCIES = ['h', 'd', 'b', 'w', 'm', 'q', 'a']
    _SEASONALITIES = ['auto', 'd', 'b', 'w', 'm', 'q', 'a']
    _SEASONALITY = {
        'h': {'auto': 24, 'd': 24, 'b': 24},
        'd': {'auto': 7, 'w': 7, 'm': 31, 'y': 365},
        'b': {'auto': 5, 'w': 5, 'm': 21, 'y': 253},
        'w': {'auto': 53, 'm': 5, 'y': 53},
        'm': {'auto': 13, 'y': 13},
        'q': {'auto': 5, 'y': 5},
        'a': {'auto': 1001}
    }
    _NOT_SEASONAL_ALIAS = ['no', 'none', 'na', 'n']
    _PERIOD_FROM_FREQ = {'a': 1, 'q': 4, 'm': 12, 'w': 52, 'd': 7, 'b': 5, 'h': 24, }
    _SEASONAL_MAX_DEFAULT = 1001
    _DEFAULT_PERIOD = 7

    def __init__(self, frequency='d', seasonality='auto', trend='auto', coef_trend=1.5):
        self.frequency = self._init_frequency(frequency)
        self.seasonality = self._init_seasonality(seasonality)
        self.seasonal = self._init_seasonal()
        self.period = self._init_period()
        self.coef_trend = coef_trend
        self.trend = self._init_trend(trend, coef_trend)

    def _init_frequency(self, frequency):
        assert frequency.lower() in self._FREQUENCIES
        return frequency.lower()

    def _init_seasonality(self, seasonality):
        is_not_seasonal = seasonality.lower() in self._NOT_SEASONAL_ALIAS
        assert (seasonality.lower() in self._SEASONALITIES) or is_not_seasonal
        return seasonality.lower()

    def _init_seasonal(self):
         return self._SEASONALITY.get(self.frequency, {}).get(self.seasonality, self._SEASONAL_MAX_DEFAULT)

    def _init_period(self):
        return self._PERIOD_FROM_FREQ.get(self.frequency, self._DEFAULT_PERIOD)

    def _init_trend(self, trend, coef_trend):

        if trend == 'auto':
            trend = nextodd(math.ceil((coef_trend * self.period) / (1 - (coef_trend / self.seasonal))))

        if (trend % 2) == 0:
            trend = nextodd(trend)

        if trend <= self.period:
            trend = nextodd(self.period + 1)

        return trend


def treat_outliers(y, lim_zscore=10, sd_prc_lo=1, sd_prc_hi=99, freq='D', coef_trend=1.5, seas=False, **kwargs):
    freq = freq[0]  # only first char
    stlparams = STLparams(frequency=freq, coef_trend=coef_trend)
    stl = STL(y, period=stlparams.period, seasonal=stlparams.seasonal, trend=stlparams.trend, robust=True)
    stl_fit = stl.fit()
    trend, seasonal, resid = stl_fit.trend, stl_fit.seasonal, stl_fit.resid
    if not seas:
        resid = seasonal + resid
        seasonal = seasonal * 0
    sd_trim = std_trim(resid, sd_prc_lo, sd_prc_hi)
    sd_iqr = (np.nanpercentile(resid, 97.5) - np.nanpercentile(resid, 2.5))/4
    sd = max(sd_trim, sd_iqr)
    resid_trim = np.maximum(-lim_zscore * sd, np.minimum(lim_zscore * sd, resid))
    y_trim = trend + seasonal + resid_trim
    return y_trim


# fig = go.Figure()
# fig.add_trace(go.Scatter(y=trend, name='trend', mode='lines+markers', opacity=0.4, line=dict(color='green', width=2)))
# fig.add_trace(go.Scatter(y=seasonal, name='seasonal', mode='lines+markers', opacity=0.4, line=dict(color='orange', width=2)))
# fig.add_trace(go.Scatter(y=resid, name='resid', mode='lines+markers', opacity=0.4, line=dict(color='red', width=2)))
# fig.add_trace(go.Scatter(y=y, name='y', mode='lines+markers', opacity=0.4, line=dict(color='black', width=2)))
# fig.add_trace(go.Scatter(y=y_trim, name='y_trim', mode='lines+markers', opacity=0.4, line=dict(color='red', width=2)))
# py.plot(fig)


def smooth_series(y, frac=0.05, it=2, **kwargs):
    if len(y) < 5:
        return y
    frac = max(frac, 3.0/float(len(y)))
    y_smooth = sm.nonparametric.lowess(endog=y, exog=np.arange(len(y)), frac=frac, it=it, return_sorted=False)
    return y_smooth


def limit_min_max(y, y_train=None, min_lim=0, max_lim=1e9, min_q=5, min_bend_q=15, max_bend_q=90, max_iqr=5.0):

    if min_q is not None and y_train is not None:
        if min_bend_q is None:
            min_bend_q = min_q + 10
        assert min_bend_q > min_q, 'min_bend_q must be above min_q'
        min_lim = np.nanpercentile(y_train, min_q)
        min_bend = np.nanpercentile(y_train, min_bend_q) + 1e-9
        min_ = np.nanmin(y)
        min_min_bend = min_ - min_bend
        min_lim_min_bend = min_lim - min_bend
        w = np.minimum(0, y - min_bend) / min_min_bend
        w = np.power(w, 0.5)
        y = np.maximum(y, min_bend + w * min_lim_min_bend)

    if min_lim is not None:
        y = np.maximum(min_lim, y)

    if max_lim is not None:
        y = np.minimum(max_lim, y)

    if max_bend_q is not None and y_train is not None:
        iqr = np.nanpercentile(y_train, 90) - np.nanpercentile(y_train, 10) + 1e-9
        max_bend = np.nanpercentile(y_train, max_bend_q)
        max_lim = max_bend + max_iqr * iqr + 1e-9
        max_ = np.nanmax(y)
        max_max_bend = max_ - max_bend
        max_lim_max_bend = max_lim - max_bend
        w = np.maximum(0, y - max_bend) / max_max_bend
        w = np.power(w, 0.5)
        y = np.minimum(y, max_bend + w * max_lim_max_bend)

    return y


def date_range_predict(train_date, steps, freq='D'):
    return pd.date_range(pd.to_datetime(train_date) + pd.DateOffset(1), periods=steps, freq=freq)
