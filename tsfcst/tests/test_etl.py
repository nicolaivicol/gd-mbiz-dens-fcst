import unittest
import pandas as pd
import os
import logging
import random
import plotly.graph_objects as go
import plotly.offline as py
import polars as pl

from etl import load_data
from utils import describe_numeric, set_display_options

set_display_options()


class TestETL(unittest.TestCase):

    def test_population(self):
        df_train, df_test, df_census, df_pop = load_data()

        assert len(df_pop['cfips'].unique()) == 3135

        print(describe_numeric(df_pop.to_pandas()))

        fig = go.Figure()
        for _ in range(100):
            cfips = random.choice(df_pop['cfips'].unique())
            tmp = df_pop.filter(pl.col('cfips') == cfips)
            fig.add_trace(
                go.Scatter(
                    x=tmp['first_day_of_month'],
                    y=tmp['population'],
                    name=str(cfips),
                    mode='lines+markers',
                    opacity=0.7
                )
            )
        fig.update_layout(title='populations')
        py.plot(fig, filename=f'temp-plot-est_pop_{id}.html')


