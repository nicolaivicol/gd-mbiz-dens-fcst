import numpy as np
import streamlit as st
import os
import random
import logging

from etl import load_data, get_ts_by_cfips
from tsfcst.params_finder import ParamsFinder
from tsfcst.models.inventory import MODELS
from tsfcst.forecasters.forecaster import Forecaster
from tsfcst.time_series import TsData
from tsfcst.utils import plot_fcsts_and_actual


log = logging.getLogger(os.path.basename(__file__))

st.set_page_config(layout="wide", page_title='Forecasting microbusiness density')


@st.cache(allow_output_mutation=True)
def get_data():
    log.debug('Loading local data files')
    df_train, df_test, df_census = load_data()
    return df_train


@st.cache(allow_output_mutation=True)
def history_selections():
    return []


df = get_data()
hs = history_selections()

# st.title("Forecasting microbusiness density")
# st.markdown("""---""")
st.sidebar.title("Control Panel")

if 'count' not in st.session_state:
    st.session_state.rand_i = 0

cfips_to_select = ([hs[-1]] if len(hs) > 0 else []) + sorted(list(np.unique(df['cfips'])))
cfips = int(st.sidebar.selectbox('Select county (by CFIPS):', cfips_to_select))

if st.sidebar.button('Select random series'):
    i = random.randint(0, len(df['cfips']) - 1)
    cfips = df['cfips'][i]
    hs.append(cfips)
    # hs.append(cfips)

target_name = st.sidebar.selectbox('Select target to forecast:', ['microbusiness_density', 'active'])
model_alias = st.sidebar.selectbox('Select model:', list(MODELS.keys()), index=list(MODELS.keys()).index('theta'))

with st.sidebar.expander('CV settings:'):
    n_train_dates = st.slider(
        label="Number of past forecasts:", value=3, min_value=1, max_value=10, step=1,
        help="How many forecasts to generate in the past for validation.",)
    periods_val = st.slider(
        label="Forecast horizon (months):", value=7, min_value=1, max_value=10, step=1,
        help="How many months ahead to forecast.",)
    step_train_dates = st.slider(
        label="Step btw forecasts (months):", value=2, min_value=1, max_value=6, step=1,
        help="How many months between the forecasts.")
    periods_test = st.slider(
        label="Periods in test (months):", value=3, min_value=0, max_value=7, step=1,
        help="How many months to use in the test fold.")
    periods_val_last = st.slider(
        label="Periods in last val fold:", value=5, min_value=1, max_value=10, step=1,
        help="How many months in the last validation fold.")
    periods_out = 7

df_ts = get_ts_by_cfips(cfips=cfips, target_name=target_name, _df=df)
ts = TsData(df_ts['first_day_of_month'], df_ts[target_name])
model_cls = MODELS[model_alias]

find_best_params = st.sidebar.button('Find best params')
using_best_placeholder = st.sidebar.empty()
# find_best_params = st.sidebar.checkbox('Find best params', False)

trend = st.sidebar.checkbox('trend', True)
seasonal = st.sidebar.checkbox('seasonal', False)
multiplicative = st.sidebar.checkbox('multiplicative', False)
level = st.sidebar.checkbox('level', True)
damp = st.sidebar.checkbox('damp', True)

n_trials = st.sidebar.select_slider(
    label="Trials:", value=100, options=[25, 50, 75, 100, 125, 150, 200],
    help="How many trials to run when searching best params.")

reg_coef = st.sidebar.select_slider(
    label="Reg coef:", value=0, options=[0, 0.01, 0.02, 0.05, 0.10, 0.5, 1, 5, 10, 20, 50, 100, 1000],
    help="Regularization coefficient.")

use_cache = st.sidebar.checkbox('use_cache', True)

fig_paralel_coords, df_trials, best_result, fig_imp, best_result_median = None, None, None, None, None

id_cache = f"{cfips}-{target_name}-{model_alias}-{n_trials}-{str(reg_coef).replace('.', '_')}"
cache_exists = ParamsFinder.cache_exists(id_cache)

if find_best_params or cache_exists:
    ParamsFinder.model_cls = model_cls
    ParamsFinder.data = ts

    ParamsFinder.trend = trend
    ParamsFinder.seasonal = seasonal
    ParamsFinder.multiplicative = multiplicative
    ParamsFinder.level = level
    ParamsFinder.damp = damp

    ParamsFinder.reg_coef = reg_coef
    ParamsFinder.n_train_dates = n_train_dates
    ParamsFinder.step_train_dates = step_train_dates
    ParamsFinder.periods_val = periods_val
    ParamsFinder.periods_test = periods_test
    ParamsFinder.periods_out = periods_out
    ParamsFinder.periods_val_last = periods_val_last

    df_trials, best_result, param_importances = ParamsFinder.find_best(
        n_trials=n_trials,
        id_cache=f"{cfips}-{target_name}-{model_alias}-{n_trials}-{str(reg_coef).replace('.', '_')}",
        use_cache=use_cache,
        parimp=True,
    )
    log.debug('best_result: \n' + str(best_result))

    best_value_median, best_params_median = ParamsFinder.best_params_top_median(df_trials)
    best_result_median = {'best_value': best_value_median, 'best_params': best_params_median}
    log.debug('best_result_median: \n' + str(best_result_median))

    top_n_trials = max(int(len(df_trials) * 0.33), 20)
    fig_paralel_coords = ParamsFinder.plot_parallel_optuna_res(df_trials.head(top_n_trials))
    fig_paralel_coords.update_layout(autosize=True, height=650, width=1200, margin=dict(l=40, r=25, t=50, b=25))
    find_best_params = False

    fig_imp = ParamsFinder.plot_importances(param_importances)

    using_best_placeholder.write(f'cfips={cfips} | using best params')
else:
    using_best_placeholder.write(f'cfips={cfips} | using default params')


if best_result is not None:
    best_params = {**best_result['best_params']}
else:
    best_params = {}

params_forecaster_names = [p['name'] for p in Forecaster.trial_params()]
params_forecaster = {k: v for k, v in best_params.items() if k in params_forecaster_names}
params_model = {k: v for k, v in best_params.items() if k not in params_forecaster_names}

fcster = Forecaster(model_cls=model_cls, data=ts, params_model=params_model, **params_forecaster)

df_fcsts_cv, metrics_cv = fcster.cv(
    n_train_dates=n_train_dates,
    step_train_dates=step_train_dates,
    periods_val=periods_val,
    periods_test=periods_test,
    periods_out=periods_out,
    periods_val_last=periods_val_last,
)

tab1, tab2, tab3, tab4 = st.tabs(["Plot forecasts", "Plot parallel", "Plot importance", "Table with trials"])

fig_fcsts = plot_fcsts_and_actual(ts.data, df_fcsts_cv)
metrics_cv_str = Forecaster.metrics_cv_str_pretty(metrics_cv)
fig_fcsts.update_layout(title=f'Forecasts by model={model_cls.__name__} for cfips={cfips}', xaxis_title=metrics_cv_str)
# py.plot(fig_fcsts)

# py.plot(fig)

# row = df.loc[df['url'] == cfips].reset_index()
# st.markdown(
#     f"[{cfips}]({cfips}) "
#     f"&nbsp;&nbsp;|&nbsp;&nbsp; Sales per day, $: {row['avg_sales_usd'][0]} "
#     f"&nbsp;&nbsp;|&nbsp;&nbsp; Orders per day: {row['avg_orders'][0]}"
# )

with tab1:
    st.plotly_chart(fig_fcsts)
    st.text(metrics_cv_str)
# st.markdown(
#     f"&nbsp| MAPE(1d): **{metrics_cv['mape']:.1f}%** "
#     f"&nbsp| MAPE(7d): **{metrics_cv['mape_7']:.1f}%** "
#     f"&nbsp| MAPE(30d): **{metrics_cv['mape_30']:.1f}%** "
#     f"&nbsp| Totals Error: **{metrics_cv['error_totals']:.1f}%** "
# )

with tab2:
    if fig_paralel_coords is not None:
        st.plotly_chart(fig_paralel_coords)
    else:
        st.text('(!) Search of best params was not yet performed.')

with tab3:
    if fig_imp is not None:
        st.plotly_chart(fig_imp)
    else:
        st.text('(!) Search of best params was not yet performed.')

with tab4:
    if df_trials is not None:
        st.markdown(f'Table 1. Trials by optuna')
        # st.text(metrics_cv_str)
        # st.text(best_result['best_params'])
        st.table(df_trials.head(100).style.set_precision(4))
    else:
        st.text('(!) Search of best params was not yet performed.')


# streamlit run app_fcsts.py --server.port 8000

# good
# 20091

