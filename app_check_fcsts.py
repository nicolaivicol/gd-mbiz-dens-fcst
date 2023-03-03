import numpy as np
import streamlit as st
import os
import random
import logging
import polars as pl
import glob

import config
from etl import load_data
from tsfcst.find_best_weights import load_best_weights
from tsfcst.predict_weights_rates import load_predicted
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
    dir_fcsts = f'{config.DIR_ARTIFACTS}/forecast_ensemble/{id_fcsts}'
    file_fcsts = f'{dir_fcsts}/fcsts_all_models.csv'
    if os.path.exists(file_fcsts):
        df_fcsts = pl.read_csv(file_fcsts, parse_dates=True)
    else:
        # load submission
        if len(glob.glob(f'{dir_fcsts}/sub*.csv')) == 1:
            file_submission = glob.glob(f'{dir_fcsts}/sub*.csv')
            df_fcsts = pl.read_csv(file_submission[0]).rename({'microbusiness_density': 'submission'})
            df_fcsts = df_fcsts \
                .with_columns(pl.col('row_id').str.split('_').alias('row_id'))\
                .with_columns(pl.col('row_id').arr.get(0).cast(pl.Int64).alias('cfips'))\
                .with_columns(pl.col('row_id').arr.get(1).str.strptime(pl.Date, fmt='%Y-%m-%d').alias('date'))
            df_fcsts = df_fcsts.join(df_pop.rename({'first_day_of_month': 'date'}), on=['cfips', 'date'])\
                .select(['cfips', 'date', 'submission', 'population'])\
                .sort(['cfips', 'date'])

    log.debug('loading best weights')
    log.debug('try loading data frame with weights per cfips')
    try:
        df_weights = load_best_weights(id_weights)
    except ValueError as e:
        try:
            df_weights = load_predicted(id_weights)
        except Exception as e:
            df_weights = None

    return df_actual, df_fcsts, df_weights


@st.cache(allow_output_mutation=True)
def history_selections():
    return []


st.sidebar.title("Control Panel")

with st.sidebar.expander('data sources:'):
    id_fcsts = st.text_input(label='id_fcsts:', value='active-naive-naive-20221201')
    id_weights = st.text_input(label='id_weights:', value='lgbm-bin-naive-ma-h025-theta-h025-folds_5-active-20220701-active-naive_ema_theta-find_best_corner-20221201')

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

target_name = st.sidebar.selectbox('Select target to forecast:', ['active', 'microbusiness_density'])

tab1, tab2 = st.tabs(['Plot forecasts', 'Details'])

df_actual_cfips = df_actual.filter(pl.col('cfips') == cfips).rename({target_name: 'actual'}).select(['date', 'actual'])
df_fcsts_cfips = df_fcsts.filter(pl.col('cfips') == cfips).drop('cfips')

models = [m for m in ['ma', 'naive', 'driftr', 'theta', 'hw', 'ensemble', 'submission'] if m in df_fcsts_cfips.columns]
if target_name == 'active':
    for m in models:
        df_fcsts_cfips = df_fcsts_cfips.with_columns((pl.col(m)/100 * pl.col('population')).alias(m))

fig_fcsts = plot_fcsts_and_actual(
    df_actual=df_actual_cfips.to_pandas(),
    df_fcsts=df_fcsts_cfips.select(['date'] + models).to_pandas(),
    target_name='actual',
    colors={'naive': 'orange', 'ma': 'brown', 'driftr': 'green', 'theta': 'blue', 'hw': 'darkblue', 'ensemble': 'red'}
)

with tab1:
    st.plotly_chart(fig_fcsts)

with tab2:
    if df_weights is not None:
        df_weights_cfips = df_weights.filter(pl.col('cfips') == cfips)
        st.table(df_weights_cfips.to_pandas())


# streamlit run app_check_fcsts.py --server.port 8002

# weights for 30031?
