# %%
'''
<b><font size="+3">EDA: GoDaddy - Microbusiness density forecasting</font></b><br><br>
<font size="+1">Nicolai Vicol (<a href="mailto:nicolaivicol@gmail.com">nicolaivicol@gmail.com</a>)</font><br>
<font size="+1">2023-02-03</font>
'''

# %%
'''
<a id="top"></a><br>
**Contents**   
- [Data](#load_data)  
- [Counts of cfips](#counts_cfips)  
- [Timeline](#timeline)    
- [Plot CFIPS for selected states - microbusiness_density and active](#plots)
    - [Alabama](#plots_alabama)
    - [California](#plots_california)
    - [Delaware](#plots_delaware)
    - [Florida](#plots_florida)
    - [Indiana](#plots_indiana)
    - [Louisiana](#plots_louisiana)
    - [Nevada](#plots_nevada)
    - [New York](#plots_new_york)
    - [Pennsylvania](#plots_pennsylvania)
    - [South Dakota](#plots_south_dakota)
    - [Texas](#plots_texas)
    - [Wisconsin](#plots_wisconsin)
    - [Wyoming](#plots_wyoming)
- [References](#references)  
'''

# %%
# load libs
import os
os.environ['NUMEXPR_MAX_THREADS'] = '8'  # to suppress a warning by numexpr
import argparse
import json
import numpy as np
import pandas as pd
import polars as pl
import logging
from tqdm.auto import tqdm
import plotly.graph_objects as go
import plotly.offline as py
import plotly.express as px
# from plotly.subplots import make_subplots
from IPython.display import display, Image


# cd at project's root folder to load modules
DIR_PROJ = 'gd-mbiz-dens-fcst'
if DIR_PROJ in os.getcwd().split('/'):
    while os.getcwd().split(os.path.sep)[-1] != DIR_PROJ:
        os.chdir('..')
    DIR_PROJ = os.getcwd()
    # print(f"working dir: {os.getcwd()}")
else:
    raise NotADirectoryError(f"please set working directory at project's root: '{DIR_PROJ}'")

# load project modules
import config
from etl import load_raw_data, load_data
from utils import (
    describe_numeric,
    display_descr_cat_freq,
    set_display_options,
    plot_aggregated_cfips,
    plot_multiple_cfips, plot_multiple_cfips_active, plot_aggregated_cfips_active, make_plots_cfips,
    plot_aggregated_cfips_population, plot_multiple_cfips_population,
)

# options
set_display_options()

log = logging.getLogger('eda.py')

# %%
'''
<a id="load_data"></a> 
## Data  
'''

# %%
df_train, df_test, df_census = load_data()
# %%

# %%
''' 
### train.csv   
- `row_id` - An ID code for the row.
- `cfips` - A unique identifier for each county using the Federal Information Processing System. 
    The first two digits correspond to the state FIPS code, while the following 3 represent the county.
- `county` - The written name of the county.
- `state` - The name of the state.
- `first_day_of_month` - The date of the first day of the month.
- `microbusiness_density` - Microbusinesses per 100 people over the age of 18 in the given county. 
    This is the **target variable**. The population figures used to calculate the density are on a two-year lag due 
    to the pace of update provided by the U.S. Census Bureau, which provides the underlying population data annually. 
    2021 density figures are calculated using 2019 population figures, etc. Not provided for the test set.
- `active` - The raw count of microbusinesses in the county. Not provided for the test set.
- `population` - The population of the county, implied from `active` and `microbusiness_density`. Not provided for the test set.
'''

# %%
display(df_train)
display(df_train.describe().to_pandas())

# %%
''' 
### test.csv   
- `row_id` - An ID code for the row.
- `cfips` - A unique identifier for each county using the Federal Information Processing System. 
    The first two digits correspond to the state FIPS code, while the following 3 represent the county.
    However there is an error in CFIPS, the leading zero is missing for some counties, where state starst with 0.
    These CFIPS have 4 digits instead of 5.
- `first_day_of_month` - The date of the first day of the month.
'''

# %%
display(df_test)
display(df_test.describe().to_pandas())

# %%
''' 
### census_starter.csv   
- `pct_bb_[year]` - The percentage of households in the county with access to broadband of any type.
- `pct_college_[year]` - The percent of the population in the county over age 25 with a 4-year college degree.
- `pct_foreign_born_[year]` - The percent of the population in the county born outside of the United States.
- `pct_it_workers_[year]` - The percent of the workforce in the county employed in information related industries.
- `median_hh_inc_[year]` - The median household income in the county.
'''

# %%
display(df_census)
display(df_census.describe().to_pandas())

# %%
''' 
### sample_submission.csv 
'''

# %%
df_ex_sub = pl.read_csv(f'{config.DIR_DATA}/sample_submission.csv')
df_ex_sub.head()

# %%
'''
<a id="counts_cfips"></a>
## Counts of `cfips`    
- **3,135** unique `cfips` in both train set and test set
'''

# %%
print(f'unique cfips in train set: {len(df_train["cfips"].unique())}')
print(f'unique cfips in test set: {len(df_test["cfips"].unique())}')
print(f'intersection of cfips in both train and test set: '
      f'{len(set(df_train["cfips"].unique()).intersection(df_test["cfips"].unique()))}')

# %%
'''
<a id="timeline"></a>
## Timeline   
[Timeline on Kaggle](https://www.kaggle.com/competitions/godaddy-microbusiness-density-forecasting/overview/timeline)   
- All the time series (3153 time series for 3153 cfips) in the train set 
    start on `2019-08` and end on `2022-10`, **39** monthly values.   
    This is the all train data we have, it won't be updated to include more during the competition.
- In the test set, time series start on `2022-11` and end on `2023-06`, **8** monthly values
- So we need to provide forecasts for the next 8 months in the submission file, while in fact the forecasts 
    will be scored against 7 months: 4 in public and 3 in private.    
  That's because the competition ends on June 14, 2023, and June data won't be available. 
  Which actually means that we need to forecast the next **7 months** after `2022-10`.
- The public part of the test data always includes only the last available month of the test data. 
  For example, in February 2023, LB is evaluated against January 2023. In March - against February.
- The deadline for submissions is March 14th 2023.
- In the private LB, as the new data comes, our prediction values will be rescored with more data available.
- On April 1st 2023 (or as soon as March data is released), our predictions will be scored against March 2023 true data;
- On May 1st 2023 (or as soon as April data is released), our predictions will be rescored using March + April 2024;
- On June 1 2023  (or as soon as May data is released, but not later than June 14), 
    a final rescore will happen using March + April + May 2023.
- June 14, 2023 - Competition End Date - Winner's announcement.
'''

# %%
fig = go.Figure()
w = 75
fig.add_trace(
    go.Scatter(
        x=['2019-08-01', '2022-10-31'], y=[2, 2],
        name='train', mode='lines', opacity=0.7, line=dict(color='black', width=w))
)
fig.add_trace(
    go.Scatter(
        x=['2022-11-01', '2022-11-30'], y=[2, 2],
        name='test: public, as of Dec 15', mode='lines', opacity=0.6, line=dict(color='red', width=w))
)
fig.add_trace(
    go.Scatter(
        x=['2022-12-01', '2022-12-31'], y=[2, 2],
        name='test: public, as of Jan', mode='lines', opacity=0.7, line=dict(color='red', width=w))
)
fig.add_trace(
    go.Scatter(
        x=['2023-01-01', '2023-01-31'], y=[2, 2],
        name='test: public, as of Feb', mode='lines', opacity=0.8, line=dict(color='red', width=w))
)
fig.add_trace(
    go.Scatter(
        x=['2023-02-01', '2023-02-28'], y=[2, 2],
        name='test: public, as of Mar 14', mode='lines', opacity=0.9, line=dict(color='red', width=w))
)
fig.add_trace(
    go.Scatter(
        x=['2023-03-01', '2023-05-31'], y=[2, 2],
        name='test: private as of Jun 14', mode='lines', opacity=0.9, line=dict(color='darkred', width=w))
)
fig.add_trace(
    go.Scatter(
        x=['2019-08-01', '2022-03-31'], y=[1, 1],
        name='local split: train', mode='lines', opacity=0.7, line=dict(color='grey', width=w))
)
fig.add_trace(
    go.Scatter(
        x=['2022-04-01', '2022-07-31'], y=[1, 1],
        name='local split: valid, less important', mode='lines', opacity=0.5, line=dict(color='red', width=w))
)
fig.add_trace(
    go.Scatter(
        x=['2022-08-01', '2022-10-31'], y=[1, 1],
        name='local split: valid, more important', mode='lines', opacity=0.7, line=dict(color='darkred', width=w))
)
fig.update_layout(title='Train/Test by Time', autosize=True, height=375,
                  legend=dict(x=1, y=0, bgcolor='rgba(0,0,0,0)'),
                  yaxis={'showticklabels': False})
py.iplot(fig)

# %%
'''
train
'''

# %%
def display_months(df):
    tmp = df \
        .groupby('cfips') \
        .agg([pl.min('first_day_of_month').alias('start_month'),
              pl.max('first_day_of_month').alias('end_month'),
              pl.count('first_day_of_month').alias('n_months')])
    display(tmp['n_months'].value_counts().to_pandas())
    display(tmp.head(3).to_pandas())

display_months(df_train)

# %%
'''
test
'''

# %%
display_months(df_test)

# %%
'''
<a id="plots"></a>
## Plot CFIPS for selected states - `microbusiness_density` and `active`
'''

# %%
'''
<a id="plots_alabama"></a>
### Alabama
'''

# %%
make_plots_cfips(df_train, 'Alabama')

# %%
'''
<a id="plots_california"></a>
### California
'''

# %%
make_plots_cfips(df_train, 'California')

# %%
'''
<a id="plots_delaware"></a>
### Delaware
'''

# %%
make_plots_cfips(df_train, 'Delaware')

# %%
'''
<a id="plots_florida"></a>
### Florida
'''

# %%
make_plots_cfips(df_train, 'Florida')

# %%
'''
<a id="plots_indiana"></a>
### Indiana
'''

# %%
make_plots_cfips(df_train, 'Indiana')

# %%
'''
<a id="plots_louisiana"></a>
### Louisiana
'''

# %%
make_plots_cfips(df_train, 'Louisiana')

# %%
'''
<a id="plots_nevada"></a>
### Nevada
'''

# %%
make_plots_cfips(df_train, 'Nevada')

# %%
'''
<a id="plots_new_york"></a>
### New York
'''

# %%
make_plots_cfips(df_train, 'New York')

# %%
'''
<a id="plots_pennsylvania"></a>
### Pennsylvania
'''

# %%
make_plots_cfips(df_train, 'Pennsylvania')


# %%
'''
<a id="plots_south_dakota"></a>
### South Dakota
'''

# %%
make_plots_cfips(df_train, 'South Dakota')

# %%
'''
<a id="plots_texas"></a>
### Texas
'''

# %%
make_plots_cfips(df_train, 'Texas')

# %%
'''
<a id="plots_wisconsin"></a>
### Wisconsin
'''

# %%
make_plots_cfips(df_train, 'Wisconsin')

# %%
'''
<a id="plots_wyoming"></a>
### Wyoming
'''

# %%
make_plots_cfips(df_train, 'Wyoming')


# %%
'''
<a id="references"></a> 
## References  
- [1] [GoDaddy - Microbusiness density forecasting](https://www.kaggle.com/competitions/godaddy-microbusiness-density-forecasting)
- [2] [GoDaddy EDA](https://www.kaggle.com/code/michaelbryantds/godaddy-eda#Distribution-of-correlations-between-microbusiness_density-and-date-seperated-by-county)
- [3] [Counties boundaries from opendatasoft.com](https://public.opendatasoft.com/explore/dataset/us-county-boundaries/table/?disjunctive.statefp&disjunctive.countyfp&disjunctive.name&disjunctive.namelsad&disjunctive.stusab&disjunctive.state_name&location=7,34.27084,-86.57227&basemap=jawg.light)
- [4] [What is public/private ratio split?](https://www.kaggle.com/competitions/godaddy-microbusiness-density-forecasting/discussion/372625)
'''


