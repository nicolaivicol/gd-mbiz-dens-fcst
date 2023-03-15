import os
import glob
from pathlib import Path
import logging
import argparse
import json
import numpy as np
import polars as pl
import pandas as pd
import lightgbm
from typing import List, Dict, Union
from sklearn.metrics import r2_score, mean_absolute_error, roc_auc_score
from datetime import date

import config
from tsfcst.find_best_weights import load_best_weights
from tsfcst.compute_feats import load_feats
from utils import describe_numeric, set_display_options

set_display_options()
log = logging.getLogger(os.path.basename(__file__))


# LightGBM
PARAMS_LGBM = {
    'objective': 'huber',  # 'huber', 'binary', # 'regression',  # 'cross_entropy',
    'boosting_type': 'gbdt',
    # 'metric': 'auc',
    'n_estimators': 1200,
    'learning_rate': 0.005,
    'max_depth': 3,  # 4,
    'num_leaves': 8,  # 15,
    'colsample_bytree': 0.25,  # aka feature_fraction
    'colsample_bynode': 0.50,
    'subsample': 0.33,  # aka bagging_fraction
    # 'bagging_freq': 1,
    'min_child_samples': 50,  # aka min_data_in_leaf
    'importance_type': 'gain',
    # 'lambda_l2': 99.5,
    'seed': 42,
}

PARAMS_LGBM_FIT = {
    'early_stopping_rounds': 100,
    'verbose': 20,
}

PARAMS_LGBM_BY_TARGET = {
    'naive': {**PARAMS_LGBM, **{'n_estimators': 800}},
    'ma': {**PARAMS_LGBM, **{'n_estimators': 800}},
    'ema': {**PARAMS_LGBM, **{'n_estimators': 800}},
    'theta': {**PARAMS_LGBM, **{'n_estimators': 800}},
    'hw': {**PARAMS_LGBM, **{'n_estimators': 1000}},
}


def load_predicted(id_prediction, folder='predict_weights_rates'):
    dir_pred_weights = f'{config.DIR_ARTIFACTS}/{folder}/{id_prediction}'
    df_weights = pl.read_csv(f'{dir_pred_weights}/predicted.csv')
    return df_weights


def feature_importance_lgbm(
        lgbm_model: lightgbm.LGBMModel,
        feature_names: List[str],
        importance_type='gain') -> pd.DataFrame:
    try:
        feat_imp = lgbm_model.feature_importance(importance_type=importance_type)
    except:
        feat_imp = lgbm_model.feature_importances_
    feat_imp = list(np.round(feat_imp / feat_imp.sum(), 4))
    feat_imp = pd.DataFrame({'feature': feature_names, 'importance': feat_imp})
    feat_imp = feat_imp.sort_values(by=['importance'], ascending=False).reset_index(drop=True)
    return feat_imp


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='test')
    ap.add_argument('--feats', default='active-20220701')
    ap.add_argument('--target_type', default='weight')
    ap.add_argument('--targets', default='active-target-naive_ema_theta-corner-20221201-20220801-manual_fix')
    ap.add_argument('--feats_full')  # e.g. 'active-20221201'
    ap.add_argument('--nfolds', default=5, type=int)
    args = ap.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    # to get weights for submission:
    # args.tag = 'full'
    # args.feats = 'active-20220701'
    # args.target_type = 'weight'
    # args.targets = 'active-target-naive_ema_theta-corner-20221201-20220801-manual_fix'
    # args.feats_full = 'active-20221201'
    # args.nfolds = 1  # use folds=1 for submission

    # to get weights for test:
    args.tag = 'test'
    args.feats = 'active-20220701'
    args.target_type = 'weight'
    args.targets = 'active-target-naive_ema_theta-corner-20221201-20220801-manual_fix'
    args.nfolds = 5  # use folds=1 for submission

    # to forecast growth rates
    # args.tag = 'full'
    # args.target_type = 'rate'
    # args.targets = 'rates_target_20220701'
    # args.feats_full = 'active-20221201'
    # args.nfolds = 1

    log.info(f'Running {os.path.basename(__file__)} with parameters: \n'
             + json.dumps(vars(args), indent=2))
    log.info('This trains model(s) to predict the best weights on test.')

    n_folds = args.nfolds

    f = args.feats_full if args.feats_full is not None else args.feats
    id_run = f"{args.tag}-{args.target_type}-folds_{n_folds}-{f}-{args.targets}"
    dir_out = f'{config.DIR_ARTIFACTS}/{Path(__file__).stem}/{id_run}'
    os.makedirs(dir_out, exist_ok=True)

    # features
    df_feats = load_feats(args.feats)
    feature_names = [f for f in df_feats.columns if f not in ['cfips', 'state']]

    try:
        df_feats_out = load_feats(args.feats_full)
        if args.target_type == 'rate':
            df_tmp = pl.DataFrame({
                'asofdate': date(2022, 12, 1),
                'date': pl.date_range(date(2023, 1, 1), date(2023, 6, 1), "1mo"),
                'horizon': [1, 2, 3, 4, 5, 6],
                'key': 1,
            })
            df_tmp = df_tmp.join(pl.DataFrame({'key': 1, 'cfips': df_feats_out['cfips']}), on='key').drop('key')
            df_feats_out = df_feats_out.join(df_tmp, on='cfips')
    except:
        df_feats_out = None

    # targets
    if args.target_type == 'weight':
        possible_target_names = ['naive', 'ma', 'ema', 'drift', 'driftr', 'theta', 'hw']
        df_targets = load_best_weights(args.targets)
    elif args.target_type == 'rate':
        possible_target_names = ['rate']
        df_targets = pl.read_csv(f'{config.DIR_ARTIFACTS}/get_rates_target/{args.targets}.csv')
        df_targets = df_targets.with_columns(pl.col('rate').clip_min(-0.025).clip_max(0.025))
        feature_names.append('horizon')
    else:
        raise ValueError(f'args.target_type={args.target_type} not recognized')

    target_names = [t for t in possible_target_names if t in df_targets.columns]
    assert len(target_names) > 0

    list_res = []
    list_feat_imps = []
    list_preds = []
    list_preds_out = []
    list_models = {}

    for target_name in target_names:
        cols_join_target = ['cfips', target_name]
        if args.target_type == 'rate':
            cols_join_target.extend(['asofdate', 'date', 'horizon'])
        df = df_targets.select(cols_join_target).rename({target_name: 'y'})
        df = df.join(df_feats, on='cfips')
        df_cfips = df.select(['cfips']).unique().sort('cfips')
        n_cfips = len(df_cfips)
        df_folds = pl.DataFrame({'fold': np.repeat(range(1, n_folds + 1), np.ceil(n_cfips/n_folds))[:n_cfips]})
        df_folds = pl.concat([df_cfips, df_folds], how='horizontal')
        df = df.join(df_folds, on='cfips', how='left')

        list_preds_folds = []

        for i_fold in range(1, n_folds+1):
            if n_folds > 1:
                df_train = df.filter(pl.col('fold') != i_fold)
                df_valid = df.filter(pl.col('fold') == i_fold)
            else:
                df_train = df.filter(pl.col('fold') == i_fold)
                df_valid = df.filter(pl.col('fold') == i_fold)

            X_train = df_train.select(feature_names).to_numpy()
            X_valid = df_valid.select(feature_names).to_numpy()
            y_train = df_train['y'].to_numpy()
            y_valid = df_valid['y'].to_numpy()
            eval_names = ['valid', 'train']
            eval_set = [(X_valid, y_valid), (X_train, y_train)]

            params_lgbm_ = PARAMS_LGBM_BY_TARGET.get(target_name, PARAMS_LGBM)
            if params_lgbm_['objective'] == 'binary':
                lgbm = lightgbm.LGBMClassifier(**params_lgbm_)
            else:
                lgbm = lightgbm.LGBMRegressor(**params_lgbm_)
            lgbm.fit(
                X=X_train,
                y=y_train,
                feature_name=feature_names,
                eval_names=eval_names,
                eval_set=eval_set,
                **PARAMS_LGBM_FIT,
            )

            if params_lgbm_['objective'] == 'binary':
                pred_y_train = lgbm.predict_proba(X_train)[:, 1]
                pred_y_valid = lgbm.predict_proba(X_valid)[:, 1]
            else:
                pred_y_train = lgbm.predict(X_train)
                pred_y_valid = lgbm.predict(X_valid)

            res = {
                'target_name': target_name,
                'fold': i_fold,
                'n_folds': n_folds,
                'best_iteration': lgbm.best_iteration_,
                'r_sq_train': r2_score(y_train, pred_y_train),
                'r_sq_valid': r2_score(y_valid, pred_y_valid),
                'mae_train': mean_absolute_error(y_train, pred_y_train),
                'mae_valid': mean_absolute_error(y_valid, pred_y_valid),
            }

            if args.target_type == 'weight':
                res['auc_train'] = roc_auc_score(y_train, pred_y_train)
                res['auc_valid'] = roc_auc_score(y_valid, pred_y_valid)
            else:
                res['auc_train'] = np.NaN
                res['auc_valid'] = np.NaN

            res = {**res, **params_lgbm_, **PARAMS_LGBM_FIT}

            df_feat_imp = feature_importance_lgbm(lgbm, feature_names)
            df_feat_imp['target_name'] = target_name
            df_feat_imp['fold'] = i_fold
            df_feat_imp = pl.from_pandas(df_feat_imp)

            if args.target_type == 'rate':
                df_pred = pl.concat(
                    [df_valid.select(['cfips', 'asofdate', 'date', 'horizon']),
                     pl.DataFrame({target_name: pred_y_valid})],
                    how='horizontal')
            else:
                df_pred = pl.DataFrame({'cfips': df_valid['cfips'], target_name: pred_y_valid})

            list_res.append(res)
            list_feat_imps.append(df_feat_imp)
            list_preds_folds.append(df_pred)

        df_pred_model = pl.concat(list_preds_folds, how='vertical')

        if df_feats_out is not None:
            X_out = df_feats_out.select(feature_names).to_numpy()
            if params_lgbm_['objective'] == 'binary':
                pred_y_out = lgbm.predict_proba(X_out)[:, 1]
            else:
                pred_y_out = lgbm.predict(X_out)

            if args.target_type == 'rate':
                df_pred_model = pl.concat(
                    [df_feats_out.select(['cfips', 'asofdate', 'date', 'horizon']),
                     pl.DataFrame({target_name: pred_y_out})],
                    how='horizontal')
            else:
                df_pred_model = pl.DataFrame({'cfips': df_feats_out['cfips'], target_name: pred_y_out})

        list_preds.append(df_pred_model)

    df_results = pl.from_records(list_res)
    df_feat_imps = pl.concat(list_feat_imps, how='vertical')
    df_predicted = list_preds[0]
    for tmp in list_preds[1:]:
        df_predicted = df_predicted.join(tmp, on='cfips')

    if args.target_type == 'weight':
        # clip negative values to 0
        for m in target_names:
            df_predicted = df_predicted.with_columns(pl.col(m).clip_min(0))

        # manually adjust predictions
        # df_predicted = df_predicted \
        #     .with_columns([
        #     (pl.col('naive') * (pl.col('naive') > 0)).alias('naive'),
        #     (pl.col('ma') * (pl.col('ma') > 0.40)).alias('ma'),
        #     (pl.col('theta') * (pl.col('theta') > 0.40)).alias('theta'),
        # ])

        # normalize weights (sum=1)
        df_predicted = df_predicted.with_columns(pl.sum(target_names).alias('sum'))
        for m in target_names:
            df_predicted = df_predicted.with_columns((pl.col(m) / pl.col('sum')).alias(m))
        df_predicted = df_predicted.drop('sum')

    df_results.write_csv(f'{dir_out}/results.csv', float_precision=4)
    df_feat_imps.write_csv(f'{dir_out}/feat_imps.csv', float_precision=4)
    df_predicted.write_csv(f'{dir_out}/predicted.csv', float_precision=4)

    log.debug(f'weights saved to: {dir_out}/predicted.csv')

    for m in target_names:
        print(f'target={m} | feature imp (top 25): \n',
              df_feat_imps
              .filter(pl.col('target_name') == m)
              .groupby(['target_name', 'feature'])
              .agg(pl.col('importance').mean())
              .sort('importance', reverse=True)
              .to_pandas()
              .head(25))

    agg = df_results.groupby('target_name')\
        .agg([pl.col('r_sq_valid').mean(),
              pl.col('mae_valid').mean(),
              pl.col('auc_valid').mean(),
              pl.col('best_iteration').mean(),
              pl.col('n_estimators').mean(),
              pl.col('learning_rate').mean(),
              pl.col('max_depth').mean(),
              pl.col('num_leaves').mean(),
              ])\
        .sort('target_name')

    print('summary results on 5 folds:: \n', agg.to_pandas())
    print('average prediction by model: \n', df_predicted.mean().to_pandas())
    if args.target_type == 'rate':
        list_stats = []
        for h in range(1, 7):
            stats = describe_numeric(df_predicted.filter(pl.col('horizon')==h).select(['rate']).to_pandas())
            stats['horizon'] = h
            list_stats.append(stats)
        stats = pd.concat(list_stats)
        print(stats)
    print('sample: \n', df_predicted.head(10).to_pandas())

# summary for weights:
# *********************************************************************************************************************
#    target_name  r_sq_valid  mae_valid  auc_valid  best_iteration  n_estimators  learning_rate  max_depth  num_leaves
# 0         ema       0.177      0.118      0.888         779.600       800.000          0.005      3.000       8.000
# 1       naive       0.299      0.340      0.823         800.000       800.000          0.005      3.000       8.000
# 2       theta       0.342      0.287      0.857         800.000       800.000          0.005      3.000       8.000
# average prediction by model:
#        cfips  naive   ema  theta
# 0 30376.038  0.641 0.074  0.285
# sample:
#     cfips  naive   ema  theta
# 0   1001  0.480 0.020  0.501
# 1   1003  0.450 0.034  0.515
# 2   1005  0.528 0.032  0.440
# 3   1007  0.591 0.320  0.089
# 4   1009  0.498 0.035  0.467
# 5   1011  0.988 0.012  0.000
# 6   1013  0.600 0.029  0.371
# 7   1015  0.524 0.039  0.436
# 8   1017  0.605 0.245  0.150
# 9   1019  0.864 0.036  0.100

# target=naive | feature imp (top 25):
#     target_name                     feature  importance
# 0        naive                   val_naive       0.107
# 1        naive               val_par_theta       0.091
# 2        naive                       avg_5       0.073
# 3        naive                    last_obs       0.073
# 4        naive  val_diff_smape_theta_naive       0.057
# 5        naive                   val_theta       0.055
# 6        naive         ratio_diff_lte10_10       0.046
# 7        naive                         avg       0.041
# 8        naive          ratio_diff_lte10_5       0.038
# 9        naive            prc_not_small_10       0.036
# 10       naive         ratio_diff_lte50_10       0.026
# 11       naive               prc_not_small       0.025
# 12       naive                  q_trend_10       0.022
# 13       naive                   sd_prc_18       0.019
# 14       naive          ratio_diff_lte5_10       0.015
# 15       naive                      sd_prc       0.015
# 16       naive                   sd_prc_10       0.015
# 17       naive                   q_trend_5       0.013
# 18       naive    val_diff_smape_ema_naive       0.013
# 19       naive          val_par_window_ema       0.013
# 20       naive                   q_trend_1       0.012
# 21       naive                smape2avg_20       0.011
# 22       naive            prc_change_lte_0       0.009
# 23       naive           ratio_diff_lte5_5       0.009
# 24       naive       pct_foreign_born_2021       0.008
# target=ema | feature imp (top 25):
#     target_name                     feature  importance
# 0          ema          val_par_window_ema       0.189
# 1          ema    val_diff_smape_ema_naive       0.186
# 2          ema                     val_ema       0.091
# 3          ema                    last_obs       0.033
# 4          ema                       avg_5       0.030
# 5          ema                   val_naive       0.026
# 6          ema                 smape2avg_5       0.025
# 7          ema                         avg       0.021
# 8          ema            prc_not_small_10       0.020
# 9          ema               prc_not_small       0.020
# 10         ema                  p_value_10       0.020
# 11         ema                  q_trend_10       0.016
# 12         ema                   val_theta       0.016
# 13         ema                     r_sq_10       0.015
# 14         ema                 smape2avg_3       0.015
# 15         ema                smape2avg_10       0.014
# 16         ema                      sd_prc       0.011
# 17         ema                smape2avg_20       0.011
# 18         ema  val_diff_smape_theta_naive       0.010
# 19         ema                     avg_prc       0.010
# 20         ema                   q_trend_5       0.009
# 21         ema                    last_prc       0.009
# 22         ema               val_par_theta       0.008
# 23         ema                    sd_prc_5       0.008
# 24         ema             pct_bb_2021_yoy       0.007
# target=theta | feature imp (top 25):
#     target_name                     feature  importance
# 0        theta                   val_theta       0.131
# 1        theta               val_par_theta       0.112
# 2        theta         ratio_diff_lte10_10       0.067
# 3        theta                    last_obs       0.065
# 4        theta  val_diff_smape_theta_naive       0.063
# 5        theta                       avg_5       0.060
# 6        theta          ratio_diff_lte10_5       0.055
# 7        theta                         avg       0.040
# 8        theta         ratio_diff_lte50_10       0.032
# 9        theta          ratio_diff_lte5_10       0.031
# 10       theta                   q_trend_5       0.023
# 11       theta                   val_naive       0.023
# 12       theta                   q_trend_1       0.020
# 13       theta                   sd_prc_18       0.018
# 14       theta           ratio_diff_lte5_5       0.018
# 15       theta            prc_not_small_10       0.015
# 16       theta                   sd_prc_10       0.015
# 17       theta                  q_trend_10       0.014
# 18       theta                      sd_prc       0.013
# 19       theta               prc_not_small       0.012
# 20       theta                  p_value_10       0.009
# 21       theta                        r_sq       0.009
# 22       theta            prc_change_lte_0       0.008
# 23       theta    val_diff_smape_ema_naive       0.007
# 24       theta         prc_change_lte_0_10       0.007

# rate
# **********************************************************************************************************************
# summary results on 5 folds::
#    target_name  r_sq_valid  mae_valid  auc_valid  best_iteration  n_estimators  learning_rate  max_depth  num_leaves
# 0        rate       0.053      0.013        NaN        1121.400      1200.000          0.005      3.000       8.000

# average prediction by model:
#        cfips asofdate  date  horizon   rate
# 0 30376.038     None  None    3.000 -0.001

#         count   mean   std    min     5%    25%    50%    75%   95%   98%   99%   max  count_nan  prc_nan  horizon
# rate 3135.000 -0.004 0.002 -0.013 -0.007 -0.005 -0.004 -0.002 0.000 0.002 0.003 0.011          0    0.000        1
# rate 3135.000 -0.002 0.002 -0.012 -0.006 -0.004 -0.002 -0.001 0.002 0.003 0.004 0.013          0    0.000        2
# rate 3135.000 -0.000 0.003 -0.012 -0.005 -0.002 -0.001  0.001 0.004 0.005 0.006 0.015          0    0.000        3
# rate 3135.000  0.000 0.003 -0.011 -0.004 -0.002 -0.000  0.002 0.005 0.006 0.007 0.015          0    0.000        4
# rate 3135.000  0.002 0.003 -0.010 -0.002  0.000  0.002  0.003 0.006 0.007 0.009 0.016          0    0.000        5

# target=rate | feature imp (top 25):
#     target_name                     feature  importance
# 0         rate                     horizon       0.187
# 1         rate                    last_prc       0.070
# 2         rate          median_hh_inc_2021       0.035
# 3         rate  val_diff_smape_theta_naive       0.028
# 4         rate         ratio_diff_lte10_10       0.022
# 5         rate                 pct_bb_2021       0.021
# 6         rate                   avg_prc_5       0.021
# 7         rate                  avg_prc_10       0.018
# 8         rate                    slope_10       0.018
# 9         rate                    last_obs       0.017
# 10        rate            pct_college_2021       0.017
# 11        rate         ratio_diff_lte50_10       0.016
# 12        rate          ratio_diff_lte10_5       0.015
# 13        rate                     hurst_5       0.015
# 14        rate                       avg_5       0.014
# 15        rate              iqr_m_ratio_10       0.014
# 16        rate                     avg_prc       0.014
# 17        rate                     r_sq_10       0.014
# 18        rate       pct_foreign_born_2021       0.014
# 19        rate                      sd_prc       0.014
# 20        rate                         avg       0.014
# 21        rate                 iqr_m_ratio       0.013
# 22        rate                    sd_prc_5       0.013
# 23        rate          ratio_diff_lte5_10       0.013
# 24        rate           chow_val_20210101       0.012
