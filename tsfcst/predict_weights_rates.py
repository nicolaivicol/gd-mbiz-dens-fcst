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

import config
from tsfcst.find_best_weights import load_best_weights
from tsfcst.compute_feats import load_feats
from utils import describe_numeric, set_display_options

set_display_options()
log = logging.getLogger(os.path.basename(__file__))


# LightGBM
PARAMS_LGBM = {
    'objective': 'regression',  # 'huber', 'binary', # 'regression',  # 'cross_entropy',
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


def load_predicted(id_prediction):
    dir_pred_weights = f'{config.DIR_ARTIFACTS}/predict_weights_rates/{id_prediction}'
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

    # to get weights for submission (uncomment):
    args.tag = 'full'
    args.feats_full = 'active-20221201'
    args.nfolds = 1  # use folds=1 for submission

    log.info(f'Running {os.path.basename(__file__)} with parameters: \n'
             + json.dumps(vars(args), indent=2))
    log.info('This trains model(s) to predict the best weights on test.')

    n_folds = args.nfolds

    id_run = f"{args.tag}-{args.target_type}-folds_{n_folds}-{args.feats}-{args.targets}"
    dir_out = f'{config.DIR_ARTIFACTS}/{Path(__file__).stem}/{id_run}'
    os.makedirs(dir_out, exist_ok=True)

    # features
    df_feats = load_feats(args.feats)
    feature_names = [f for f in df_feats.columns if f not in ['cfips', 'state']]

    try:
        df_feats_out = load_feats(args.feats_full)
    except:
        df_feats_out = None

    # targets
    if args.target_type == 'weight':
        possible_target_names = ['naive', 'ma', 'ema', 'drift', 'driftr', 'theta', 'hw']
        df_targets = load_best_weights(args.targets)
    elif args.target_type == 'rate':
        possible_target_names = ['rate']
        df_targets = None
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
        df = df_targets.select(['cfips', target_name]).rename({target_name: 'y'})
        df = df.join(df_feats, on='cfips')
        df_folds = pl.DataFrame({'fold': np.repeat(range(1, n_folds + 1), np.ceil(len(df)/n_folds))[:len(df)]})
        df = pl.concat([df, df_folds], how='horizontal')

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
                'auc_train': roc_auc_score(y_train, pred_y_train),
                'auc_valid': roc_auc_score(y_valid, pred_y_valid),
            }
            res = {**res, **params_lgbm_, **PARAMS_LGBM_FIT}

            df_feat_imp = feature_importance_lgbm(lgbm, feature_names)
            df_feat_imp['target_name'] = target_name
            df_feat_imp['fold'] = i_fold
            df_feat_imp = pl.from_pandas(df_feat_imp)

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
    print('sample: \n', df_predicted.head(10).to_pandas())


# summary results on 5 folds:
#    target_name  r_sq_valid  mae_valid  auc_valid  best_iteration  n_estimators  learning_rate  max_depth  num_leaves
# 0         ema       0.300      0.148      0.915         800.000       800.000          0.005      3.000       8.000
# 1       naive       0.309      0.361      0.820         800.000       800.000          0.005      3.000       8.000
# 2       theta       0.382      0.291      0.857         800.000       800.000          0.005      3.000       8.000

# average prediction by model:
#        cfips  naive   ema  theta
# 0 30376.038  0.548 0.109  0.343

# sample:
#     cfips  naive   ema  theta
# 0   1001  0.403 0.006  0.590
# 1   1003  0.124 0.000  0.876
# 2   1005  0.469 0.059  0.472
# 3   1007  0.668 0.200  0.131
# 4   1009  0.437 0.041  0.522
# 5   1011  0.687 0.245  0.068
# 6   1013  0.419 0.243  0.338
# 7   1015  0.871 0.010  0.118
# 8   1017  0.599 0.008  0.393
# 9   1019  0.563 0.263  0.175

# target=naive | feature imp:
#     target_name                     feature  importance
# 0        naive               val_par_theta       0.162
# 1        naive                   val_naive       0.152
# 2        naive  val_diff_smape_theta_naive       0.086
# 3        naive                   val_theta       0.046
# 4        naive         ratio_diff_lte10_10       0.037
# 5        naive                       avg_5       0.034
# 6        naive                    last_obs       0.030
# 7        naive         ratio_diff_lte50_10       0.030
# 8        naive                         avg       0.028
# 9        naive          val_par_window_ema       0.028
# 10       naive          ratio_diff_lte5_10       0.027
# 11       naive          ratio_diff_lte10_5       0.026
# 12       naive    val_diff_smape_ema_naive       0.026
# 13       naive                     val_ema       0.025
# 14       naive                  q_trend_10       0.024
# 15       naive                   q_trend_5       0.020

# target=ema | feature imp:
#     target_name                     feature  importance
# 0          ema          val_par_window_ema       0.210
# 1          ema    val_diff_smape_ema_naive       0.196
# 2          ema                     val_ema       0.158
# 3          ema                   val_naive       0.038
# 4          ema                  q_trend_10       0.024
# 5          ema                     r_sq_10       0.022
# 6          ema                  p_value_10       0.021
# 7          ema                 smape2avg_3       0.020
# 8          ema                   val_theta       0.017
# 9          ema                 smape2avg_5       0.016
# 10         ema                  avg_prc_10       0.016
# 11         ema                       slope       0.014
# 12         ema         ratio_diff_lte10_10       0.013
# 13         ema          ratio_diff_lte10_5       0.012
# 14         ema                   q_trend_5       0.012
# 15         ema          ratio_diff_lte5_10       0.010

# target=theta | feature imp:
#     target_name                     feature  importance
# 0        theta               val_par_theta       0.194
# 1        theta                   val_theta       0.138
# 2        theta  val_diff_smape_theta_naive       0.083
# 3        theta         ratio_diff_lte10_10       0.051
# 4        theta                       avg_5       0.045
# 5        theta          ratio_diff_lte5_10       0.043
# 6        theta                    last_obs       0.042
# 7        theta         ratio_diff_lte50_10       0.037
# 8        theta          ratio_diff_lte10_5       0.036
# 9        theta                         avg       0.035
# 10       theta                   val_naive       0.029
# 11       theta                   q_trend_5       0.025
# 12       theta          val_par_window_ema       0.019
# 13       theta                   q_trend_1       0.017
# 14       theta                   sd_prc_18       0.012
# 15       theta           ratio_diff_lte5_5       0.012
