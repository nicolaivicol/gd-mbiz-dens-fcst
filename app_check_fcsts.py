import numpy as np
import streamlit as st
import os
import random
import logging
import polars as pl
import glob

import config
from etl import load_data, get_df_ts_by_cfips
from tsfcst.params_finder import ParamsFinder
from tsfcst.models.inventory import MODELS
from tsfcst.forecasters.forecaster import Forecaster
from tsfcst.time_series import TsData
from tsfcst.utils_tsfcst import plot_fcsts_and_actual


log = logging.getLogger(os.path.basename(__file__))

st.set_page_config(layout="wide", page_title='Check Forecasts')


@st.cache(allow_output_mutation=True)
def get_data(id_fcsts, id_weights):
    log.debug('Loading local data files')
    log.debug('loading actual')
    df_actual, _, _, df_pop = load_data()
    df_actual = df_actual.rename({'first_day_of_month': 'date'})

    log.debug('loading forecasts')
    dir_fcsts = f'{config.DIR_ARTIFACTS}/forecast_ensemble'
    df_fcsts = pl.read_csv(f'{dir_fcsts}/{id_fcsts}/fcsts_all_models.csv', parse_dates=True)\
        .with_columns(pl.col('cfips').cast(pl.Int32))

    log.debug('loading best weights')
    dir_best_weights = f'{config.DIR_ARTIFACTS}/find_best_weights/{id_weights}'
    files_best_weights = glob.glob(f'{dir_best_weights}/*.csv')
    df_weights = pl.concat([pl.read_csv(f) for f in files_best_weights])

    return df_actual, df_fcsts, df_weights


@st.cache(allow_output_mutation=True)
def history_selections():
    return []


st.sidebar.title("Control Panel")

with st.sidebar.expander('data sources:'):
    id_fcsts = st.text_input(label='id_fcsts:', value='microbusiness_density-test-20220701')
    id_weights = st.text_input(label='id_weights:', value='microbusiness_density-cv-20220701')

df_actual, df_fcsts, df_weights = get_data(id_fcsts, id_weights)
hs = history_selections()

if 'count' not in st.session_state:
    st.session_state.rand_i = 0

list_cfips = sorted(list(np.unique(df_fcsts['cfips'])))
cfips_to_select = ([hs[-1]] if len(hs) > 0 else []) + list_cfips
cfips = int(st.sidebar.selectbox('Select county (by CFIPS):', cfips_to_select))

select_random = st.sidebar.checkbox('select random', value=True)

if st.sidebar.button('Select next / random'):
    if select_random:
        i = random.randint(0, len(list_cfips) - 1)
    else:
        i = min(list_cfips.index(cfips) + 1, len(list_cfips) - 1)

    cfips = list_cfips[i]
    hs.append(cfips)

st.sidebar.text(f'selected cfips: {cfips}')

target_name = st.sidebar.selectbox('Select target to forecast:', ['microbusiness_density', 'active'])

tab1, tab2 = st.tabs(['Plot forecasts', 'Details'])

df_actual_cfips = df_actual.filter(pl.col('cfips') == cfips).rename({target_name: 'actual'}).select(['date', 'actual'])
df_fcsts_cfips = df_fcsts.filter(pl.col('cfips') == cfips).drop('cfips')

fig_fcsts = plot_fcsts_and_actual(
    df_actual=df_actual_cfips.to_pandas(),
    df_fcsts=df_fcsts_cfips.to_pandas(),
    target_name='actual',
    colors={'naive': 'orange', 'ma': 'brown', 'theta': 'blue', 'hw': 'darkblue', 'ensemble': 'red'}
)

df_weights_cfips = df_weights.filter(pl.col('cfips') == cfips)

with tab1:
    st.plotly_chart(fig_fcsts)

with tab2:
    st.table(df_weights_cfips.to_pandas())


# streamlit run app_check_fcsts.py --server.port 8003

# weights for 30031?
