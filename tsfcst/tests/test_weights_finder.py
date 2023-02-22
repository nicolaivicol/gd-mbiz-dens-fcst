import io
import unittest
import polars as pl
import json
import time

from tsfcst.tests.data_samples import FCSTS_CV_NAIVE, FCSTS_CV_MA, FCSTS_CV_THETA, FCSTS_CV_HW, ACTUAL
from tsfcst.weights_finder import WeightsFinder
from utils import set_display_options

set_display_options()


class TestWeightsFinder(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.fcsts = {
            'naive': FCSTS_CV_NAIVE,
            'ma': FCSTS_CV_MA,
            'theta': FCSTS_CV_THETA,
            'hw': FCSTS_CV_HW,
        }

        df_fcsts = None
        for model_name, data_str_csv in self.fcsts.items():
            df = pl.read_csv(io.StringIO(data_str_csv))
            cols_fcsts = [col for col in df.columns if col.startswith('fcst_')]
            df = pl.concat([df.select(['date', col_fcst]).rename({col_fcst: model_name}).with_columns(pl.lit(col_fcst).alias('fold'))
                            for col_fcst in cols_fcsts])
            df = df.filter(pl.col(model_name).is_not_null())
            # df = pl.DataFrame({'date': df['date'], model_name: df[cols_fcsts].mean(axis=1)})

            if df_fcsts is None:
                df_fcsts = df
            else:
                df_fcsts = df_fcsts.join(df, on=['date', 'fold'])

        df_fcsts = df_fcsts.drop('fold')

        df_actual = pl.read_csv(io.StringIO(ACTUAL)).select(['first_day_of_month', 'microbusiness_density'])
        df = df_actual.join(df_fcsts, left_on='first_day_of_month', right_on='date', how='inner')
        df = df.filter(pl.col('first_day_of_month') <= '2022-10-01')

        self.y_preds = df[list(self.fcsts.keys())].to_numpy()
        self.y_true = df['microbusiness_density'].to_numpy()

    def test_find_best(self):
        WeightsFinder.y_true = self.y_true
        WeightsFinder.y_preds = self.y_preds
        WeightsFinder.model_names = list(self.fcsts.keys())

        tic = time.time()
        res = WeightsFinder.find_best()

        print(f'time elapsed: {time.strftime("%Hh %Mmin %Ssec", time.gmtime(time.time() - tic))}')
        print('best weights: ', res['best_params'])
        print('smape: \n',
              json.dumps({
                  'best': WeightsFinder.smape(res['weights']),
                  'naive': WeightsFinder.smape([1, 0, 0, 0]),
                  'ma': WeightsFinder.smape([0, 1, 0, 0]),
                  'theta': WeightsFinder.smape([0, 0, 1, 0]),
                  'hw': WeightsFinder.smape([0, 0, 0, 1]),
              }, indent=2))
        print('10 best trials:\n', res['study'].trials_dataframe().sort_values('value').head(10))
        print('predefined trials:\n', res['study'].trials_dataframe().head(8))

    def test_from_errors(self):
        WeightsFinder.y_true = self.y_true
        WeightsFinder.y_preds = self.y_preds
        WeightsFinder.model_names = list(self.fcsts.keys())

        tic = time.time()
        res = WeightsFinder.find_from_errors()

        print(f'time elapsed: {time.strftime("%Hh %Mmin %Ssec", time.gmtime(time.time() - tic))}')
        print('best weights: ', res['best_params'])
        print('smape: \n',
              json.dumps({
                  'best': WeightsFinder.smape(res['weights']),
                  'naive': WeightsFinder.smape([1, 0, 0, 0]),
                  'ma': WeightsFinder.smape([0, 1, 0, 0]),
                  'theta': WeightsFinder.smape([0, 0, 1, 0]),
                  'hw': WeightsFinder.smape([0, 0, 0, 1]),
              }, indent=2))
