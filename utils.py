import os
import shutil
import math
import numpy as np
import pandas as pd
import polars as pl
from typing import List
import warnings

from IPython.display import display, HTML
import plotly.graph_objects as go
import plotly.offline as py
from plotly.subplots import make_subplots

import config

HEIGHT_PLOT = 650


def describe_numeric(df, cols_num=None, percentiles=None, stats_nans=True):
    """
    Describe numeric columns
    :param df: pandas data frame
    :param cols_num: numeric columns to describe, by default: identified automatically
    :param percentiles: percentiles to compute, default: [0.05, 0.25, 0.50, 0.75, 0.95]
    :return: pandas df with stats
    """
    if cols_num is None:
        cols_num = list(df.head(1).select_dtypes(include=['number']).columns)
    if percentiles is None:
        percentiles = [0.05, 0.25, 0.50, 0.75, 0.95, 0.98, 0.99]
    if len(cols_num) == 0:
        return None
    d_describe = df[cols_num].describe(percentiles=percentiles).T
    if stats_nans:
        d_describe['count_nan'] = df.isnull().sum()
        d_describe['prc_nan'] = 1 - d_describe['count'] / float(df.shape[0])
    return d_describe


def describe_categorical(df, cols=None):
    """
    Describe categorical columns
    :param df: pandas data frame
    :param cols: categorical columns to describe, by default: identified automatically
    :return: pandas df with stats
    """
    if cols is None:
        cols = list(df.head(1).select_dtypes(include=['object']).columns)
    if len(cols) == 0:
        return None
    d_describe = df[cols].astype('category').describe().T
    return d_describe


def describe_categorical_freq(x: pd.Series, name: str = None, max_show: int = 10, min_prc: float = 0.001):
    """
    Describe series with categorical values (counts, frequency)
    :param x: series to describe
    :param name: name
    :param max_show: max values to show
    :param min_prc: minimum size (in %) for the category to show in stats
    :return: pandas df with stats
    """
    if name is None:
        try:
            name = x.name
        except:
            name = 'value'
    tmp = pd.DataFrame({name: x})

    agg = tmp.groupby([name], dropna=False, as_index=True).agg({name: len}).rename(columns={name: 'count'})
    agg['percentage'] = agg['count'] / sum(agg['count'])
    agg.sort_values(['count'], ascending=False, inplace=True)
    agg.reset_index(drop=False, inplace=True)
    filter_out = (((agg['percentage'] < min_prc)
                   & (pd.Series(range(len(agg))) > max_show))
                  | (pd.Series(range(len(agg))) > max_show))
    agg = agg.loc[~filter_out, ]
    return agg


def display_descr_cat_freq(df, cols=None, skip_freq_cols=None, show_title=False):
    """
    Describe categorical columns in dataframe (counts, frequency)
    :param df: data frame
    :param cols: for which columns to compute statistics, by default: identifed automatically
    :param skip_freq_cols: which columns to skip
    :return: pandas df with stats
    """
    if cols is None:
        cols = list(df.head(1).select_dtypes(include=['object']).columns)
    if skip_freq_cols is None:
        skip_freq_cols = []
    if len(cols) == 0:
        return None
    display(describe_categorical(df, cols))
    for col in cols:
        if col not in skip_freq_cols:
            if show_title:
                display(HTML(f'<br><b>{col}</b>'))
            # else:
            #     display(HTML('<br>'))
            display(describe_categorical_freq(df[col]))


def set_display_options():
    """
    Set display options for numbers, table width, etc.
    :return: None
    """
    pd.set_option('plotting.backend', 'plotly')
    pd.set_option('display.max_rows', 100)
    pd.set_option('display.max_columns', 50)
    pd.set_option('display.width', 2000)
    pd.set_option('display.max_colwidth', 150)
    pd.set_option('max_colwidth', 150)
    pd.set_option('display.precision', 2)
    pd.set_option('display.chop_threshold', 1e-6)
    # pd.set_option('expand_frame_repr', True)
    pd.set_option('display.float_format', lambda x: '%.3f' % x)
    warnings.simplefilter('ignore')
    pl.Config.set_tbl_rows(10)
    display(HTML("<style>.container { width:80% !important; }</style>"))


def get_last_commit_hash():
    try:
        import subprocess
        result = subprocess.check_output(['git', 'log', '-1', '--pretty=format:"%H"'])
        return result.decode('utf-8').replace('"', '')[:8]
    except Exception as e:
        return None


def get_timestamp():
    from datetime import datetime
    return datetime.now().strftime("%Y%m%d%H%M%S")


def get_submit_file_name(prefix='submission', tag=None):
    tag = '' if tag is None else f'-{tag}'
    commit_hash = '' if get_last_commit_hash() is None else f'-{get_last_commit_hash()}'
    timestamp = f'-{get_timestamp()}'
    return f'{prefix}{timestamp}{tag}{commit_hash}'


def get_best_metric(lgbm_ranker):
    try:
        metric_, best_score = list(lgbm_ranker.best_score_['valid'].items())[0]
    except (AttributeError, IndexError):
        try:
            metric_, best_score = list(lgbm_ranker.best_score_['train'].items())[0]
        except:
            metric_, best_score = 'NA', 'NA'

    return metric_, best_score


def get_best_iter(lgbm_ranker):
    best_iter = lgbm_ranker.best_iteration_ \
        if lgbm_ranker.best_iteration_ is not None \
        else lgbm_ranker.get_params().get('n_estimators')
    return best_iter


def plot_forecast_in_out(self):
    fig = go.Figure()
    t = self.forecaster.target_name
    fig.add_trace(
        go.Scatter(
            x=pd.to_datetime(np.concatenate([self.forecast_in['upgrade'], self.forecast_out['upgrade']])),
            y=np.concatenate([self.forecast_in[t], self.forecast_out[t]]),
            name=f"actual {t}", mode='lines', opacity=0.7,
            line=dict(color='black', width=2))
    )
    fig.add_trace(
        go.Scatter(
            x=self.forecast_in['upgrade'], y=self.forecast_in[f'pred_{t}'],
            name=f"forecast in", mode='lines', opacity=0.7, line=dict(color='green', width=2))
    )
    fig.add_trace(
        go.Scatter(
            x=self.forecast_out['upgrade'], y=self.forecast_out[f'pred_{t}'],
            name=f"forecast out", mode='lines', opacity=0.7, line=dict(color='red', width=2))
    )
    fig.update_layout(title=self.forecaster_class.__name__, autosize=True, height=750,
                      legend=dict(x=0, y=1, bgcolor='rgba(0,0,0,0)'))
    py.iplot(fig)
    # py.plot(fig)


def plot_multiple_cfips(df, measure='microbusiness_density', title=None, max_n=25, height=config.HEIGHT_PLOT_MEDIUM):
    fig = go.Figure()

    if title is None:
        title = ', '.join(sorted(list(df['state'].unique())))

    for cfips in sorted(list(df['cfips'].unique()))[:max_n]:
        df_cfips = df.filter(pl.col('cfips') == cfips)
        fig.add_trace(
            go.Scatter(
                x=df_cfips['first_day_of_month'],
                y=df_cfips[measure],
                name=cfips,
                mode='lines',
                opacity=0.7
            )
        )
    fig.update_layout(
        title=f'{title} - {measure}',
        autosize=True,
        height=height,
        legend=dict(x=1, y=0, bgcolor='rgba(0,0,0,0)'),
        yaxis={'title': measure},
        margin=config.PLOT_MARGINS_MEDIUM,
    )
    py.iplot(fig)


def plot_multiple_cfips_microbiz_dens(df):
    return plot_multiple_cfips(df, measure='microbusiness_density')


def plot_multiple_cfips_active(df):
    return plot_multiple_cfips(df, measure='active')


def plot_multiple_cfips_population(df):
    return plot_multiple_cfips(df, measure='population')


def plot_aggregated_cfips(df, title=None, measure='microbusiness_density', by='first_day_of_month',
                          lo_q=0.25, mid='mean', hi_q=0.75, include_hi_lo=True, height=config.HEIGHT_PLOT_LOW):

    if title is None:
        title = ', '.join(sorted(list(df['state'].unique())))

    df_agg = df \
        .groupby(by) \
        .agg([pl.quantile(measure, hi_q).alias(f'q{hi_q * 100}'),
              pl.median(measure).alias('median'),
              pl.mean(measure).alias('mean'),
              pl.sum(measure).alias('sum'),
              pl.quantile(measure, lo_q).alias(f'q{lo_q * 100}'),
              ]) \
        .sort(by)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            name=mid,
            x=df_agg['first_day_of_month'],
            y=df_agg[mid],
            mode='lines',
            line=dict(color='rgb(31, 119, 180)', width=3),
            showlegend=False
        )
    )

    if include_hi_lo:
        fig.add_trace(
            go.Scatter(
                name=f'q{hi_q * 100}',
                x=df_agg['first_day_of_month'],
                y=df_agg[f'q{hi_q * 100}'],
                mode='lines',
                marker=dict(color="#444"),
                line=dict(width=0),
                showlegend=False
            )
        )

        fig.add_trace(
            go.Scatter(
                name=f'q{lo_q * 100}',
                x=df_agg['first_day_of_month'],
                y=df_agg[f'q{lo_q * 100}'],
                marker=dict(color="#444"),
                line=dict(width=0),
                mode='lines',
                fillcolor='rgba(68, 68, 68, 0.3)',
                fill='tonexty',
                showlegend=False
            )
        )

    fig.update_layout(
        title=f'{title} - {measure}',
        autosize=True,
        height=height,
        yaxis_title=measure,
        hovermode="x",
        margin=config.PLOT_MARGINS_SMALL,
    )
    py.iplot(fig)


def plot_aggregated_cfips_microbiz_dens(df):
    return plot_aggregated_cfips(df, measure='microbusiness_density')


def plot_aggregated_cfips_active(df):
    return plot_aggregated_cfips(df, measure='active')


def plot_aggregated_cfips_population(df):
    return plot_aggregated_cfips(df,  measure='population', mid='sum', include_hi_lo=False)


def make_plots_cfips(df_train, state):
    if config.MAKE_PLOTS:
        return

    plot_multiple_cfips_microbiz_dens(df_train.filter(pl.col('state') == state))
    plot_multiple_cfips_active(df_train.filter(pl.col('state') == state))
    plot_multiple_cfips_population(df_train.filter(pl.col('state') == state))

    plot_aggregated_cfips_microbiz_dens(df_train.filter(pl.col('state') == state))
    plot_aggregated_cfips_active(df_train.filter(pl.col('state') == state))
    plot_aggregated_cfips_population(df_train.filter(pl.col('state') == state))


def get_stats(values, by, sub_na_with=None):
    if sub_na_with is not None:
        by = np.array(by)
        by[~(by == by)] = sub_na_with
    d = pd.DataFrame({'values': np.array(values).astype('float'), 'by': np.array(by)})
    d_agg = d.groupby('by', as_index=False).agg(
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
    )
    return d_agg


def get_box_chart(x, y, name=None, return_stats=False, order_by_count=False, min_prc_count=None, sub_na_with=None, **kwargs):
    d_agg = get_stats(y, x, sub_na_with)

    if order_by_count:
        d_agg = d_agg.sort_values(['count'], ascending=False)

    if min_prc_count is not None:
        d_agg = d_agg.loc[d_agg['count'] > (min_prc_count*sum(d_agg['count']))]

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
    if return_stats:
        return box, d_agg
    else:
        return box


def plot_box_plot(target_values, by_values, yaxis_title='value', xaxis_title='by', x_as_category=True,
                  order_by_count=False, min_prc_count=None, sub_na_with=None):
    fig = go.Figure()
    trace_ = get_box_chart(x=by_values, y=target_values, name=xaxis_title,
                           order_by_count=order_by_count, min_prc_count=min_prc_count, sub_na_with=sub_na_with)
    fig.add_trace(trace_)
    fig.update_layout(
        title='',
        autosize=True, legend=dict(x=0, y=1, bgcolor='rgba(0,0,0,0)'), height=config.HEIGHT_PLOT_MEDIUM,  # width=1200,
        margin=dict(l=25, r=25, t=25, b=25), # yaxis_range=rng_y_boxplot, # boxmode='group',
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
    )
    if x_as_category:
        fig.update_xaxes(type='category')

    return fig
