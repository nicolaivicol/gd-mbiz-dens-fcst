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
    'objective': 'binary', # 'regression',  # 'cross_entropy',
    'boosting_type': 'gbdt',
    # 'metric': 'auc',
    'n_estimators': 600,
    'learning_rate': 0.01,
    'max_depth': 3,  # 4,
    'num_leaves': 8,  # 15,
    'colsample_bytree': 0.25,  # aka feature_fraction
    'subsample': 0.33,  # aka bagging_fraction
    # 'bagging_freq': 1,
    'min_child_samples': 50,  # aka min_data_in_leaf
    'importance_type': 'gain',
    'seed': 42,
}

PARAMS_LGBM_FIT = {
    'early_stopping_rounds': 100,
    'verbose': 20,
}

PARAMS_LGBM_BY_TARGET = {
    'naive': {**PARAMS_LGBM, **{'n_estimators': 400}},
    'ma': {**PARAMS_LGBM, **{'n_estimators': 400}},
    'theta': {**PARAMS_LGBM, **{'n_estimators': 400}},
    'hw': {**PARAMS_LGBM, **{'n_estimators': 500}},
}


def load_predicted_weights(weights_id, model_names = None):
    dir_pred_weights = f'{config.DIR_ARTIFACTS}/predict_best_weights_with_model/{weights_id}'
    df_weights = pl.read_csv(f'{dir_pred_weights}/predicted_weights.csv')

    # # reweight to have sum of weights = 1:
    # if model_names is None:
    #     model_names = ['naive', 'ma', 'theta', 'hw']
    # df_weights = df_weights.with_columns(pl.sum(model_names).alias('sum_weights'))
    # for model_name in model_names:
    #     df_weights = df_weights.with_columns(pl.col(model_name) / pl.col('sum_weights'))
    # df_weights = df_weights.drop('sum_weights')

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
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--tag', default='lgbm-bin-naive-theta-h040')
    parser.add_argument('-f', '--feats', default='active-20220701', help='time series features as of before test')
    parser.add_argument('-w', '--weights_cv', help='best weights on cv',
                        default='active-cv-naive_ema_theta-find_best_corner-20220701')
    parser.add_argument('-v', '--weights_test', help='best weights on test (target variable)',
                        default='active-test-naive_ema_theta-find_best_corner-20221201')

    # uncomment for submission
    parser.add_argument('--feats_full', default='active-20221201')
    parser.add_argument('--weights_cv_full', default='active-full-naive_ema_theta-find_best_corner-20221201')

    # use folds=1 for submission
    parser.add_argument('-n', '--nfolds', default=1, type=int)
    args = parser.parse_args()
    return args


def load_all_feats(id_feats, id_weights_cv):
    df_feats = load_feats(id_feats)

    df_weights_cv = load_best_weights(id_weights_cv)
    df_weights_cv = df_weights_cv.with_columns([
        (pl.col('smape_ma') - pl.col('smape_naive')).alias('diff_smape_ma_naive'),
        (pl.col('smape_theta') - pl.col('smape_naive')).alias('diff_smape_theta_naive')
    ])
    map_weights_cv_names = {c: f'val_{c}' for c in df_weights_cv.columns if c != 'cfips'}
    df_weights_cv = df_weights_cv.rename(map_weights_cv_names).sort('cfips')
    df_feats = df_feats.join(df_weights_cv, on='cfips')
    return df_feats


if __name__ == '__main__':
    args = parse_args()

    log.info(f'Running {os.path.basename(__file__)} with parameters: \n'
             + json.dumps(vars(args), indent=2))
    log.info('This trains model(s) to predict the best weights on test.')

    n_folds = args.nfolds
    model_names = ['naive', 'ma', 'theta']

    id_run = f"{args.tag}-folds_{n_folds}-{args.feats}-{args.weights_test}"
    dir_out = f'{config.DIR_ARTIFACTS}/{Path(__file__).stem}/{id_run}'
    os.makedirs(dir_out, exist_ok=True)

    # features
    df_feats = load_all_feats(args.feats, args.weights_cv)
    feature_names = [f for f in df_feats.columns if f not in ['cfips']]

    try:
        df_feats_out = load_all_feats(args.feats_full, args.weights_cv_full)
    except:
        df_feats_out = None

    # targets
    df_weights_test = load_best_weights(args.weights_test).select(['cfips'] + model_names).sort('cfips')

    list_res = []
    list_feat_imps = []
    list_preds = []
    list_preds_out = []
    list_models = {}

    for model_name in model_names:

        df = df_weights_test.select(['cfips', model_name]).rename({model_name: 'y'})
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

            params_lgbm_ = PARAMS_LGBM_BY_TARGET.get(model_name, PARAMS_LGBM)
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
                'target_name': model_name,
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
            df_feat_imp['target_name'] = model_name
            df_feat_imp['fold'] = i_fold
            df_feat_imp = pl.from_pandas(df_feat_imp)

            df_pred = pl.DataFrame({'cfips': df_valid['cfips'], model_name: pred_y_valid})

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
            df_pred_model = pl.DataFrame({'cfips': df_feats_out['cfips'], model_name: pred_y_out})

        list_preds.append(df_pred_model)

    df_results = pl.from_records(list_res)
    df_feat_imps = pl.concat(list_feat_imps, how='vertical')
    # df_preds = pl.concat(list_preds, how='horizontal')
    df_predicted_weights = list_preds[0]
    for tmp in list_preds[1:]:
        df_predicted_weights = df_predicted_weights.join(tmp, on='cfips')

    # clip negative values to 0
    for m in model_names:
        df_predicted_weights = df_predicted_weights.with_columns(pl.col(m).clip_min(0))

    # manually adjust predictions
    df_predicted_weights = df_predicted_weights \
        .with_columns([
        (pl.col('naive') * (pl.col('naive') > 0)).alias('naive'),
        (pl.col('ma') * (pl.col('ma') > 0.40)).alias('ma'),
        (pl.col('theta') * (pl.col('theta') > 0.40)).alias('theta'),
    ])

    # normalize weights (sum=1)
    df_predicted_weights = df_predicted_weights.with_columns(pl.sum(model_names).alias('sum'))
    for m in model_names:
        df_predicted_weights = df_predicted_weights.with_columns((pl.col(m)/pl.col('sum')).alias(m))
    df_predicted_weights = df_predicted_weights.drop('sum')

    df_results.write_csv(f'{dir_out}/results.csv', float_precision=4)
    df_feat_imps.write_csv(f'{dir_out}/feat_imps.csv', float_precision=4)
    df_predicted_weights.write_csv(f'{dir_out}/predicted_weights.csv', float_precision=4)

    log.debug(f'weights saved to: {dir_out}/predicted_weights.csv')

    for m in model_names:
        print(df_feat_imps
              .filter(pl.col('target_name') == m)
              .groupby(['target_name', 'feature'])
              .agg(pl.col('importance').mean())
              .sort('importance', reverse=True)
              .to_pandas())

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
    print(agg)

    print(df_predicted_weights.mean())

    print(df_predicted_weights.head(10).to_pandas())



# summary results on 5 folds
# ┌───────────┬──────────┬───────────┬────────────┬────────────┬────────────┬───────────┬────────────┐
# │ target_na ┆ r_sq_val ┆ mae_valid ┆ best_itera ┆ n_estimato ┆ learning_r ┆ max_depth ┆ num_leaves │
# │ me        ┆ id       ┆ ---       ┆ tion       ┆ rs         ┆ ate        ┆ ---       ┆ ---        │
# │ ---       ┆ ---      ┆ f64       ┆ ---        ┆ ---        ┆ ---        ┆ f64       ┆ f64        │
# │ str       ┆ f64      ┆           ┆ f64        ┆ f64        ┆ f64        ┆           ┆            │
# ╞═══════════╪══════════╪═══════════╪════════════╪════════════╪════════════╪═══════════╪════════════╡
# │ ma        ┆ 0.225224 ┆ 0.15436   ┆ 388.6      ┆ 600.0      ┆ 0.01       ┆ 3.0       ┆ 8.0        │
# │ naive     ┆ 0.105075 ┆ 0.431547  ┆ 367.6      ┆ 400.0      ┆ 0.01       ┆ 3.0       ┆ 8.0        │
# │ theta     ┆ 0.115362 ┆ 0.362063  ┆ 333.8      ┆ 400.0      ┆ 0.01       ┆ 3.0       ┆ 8.0        │
# └───────────┴──────────┴───────────┴────────────┴────────────┴────────────┴───────────┴────────────┘

# distribution
# ┌─────────────┬──────────┬──────────┬──────────┐
# │ cfips       ┆ naive    ┆ ma       ┆ theta    │
# │ ---         ┆ ---      ┆ ---      ┆ ---      │
# │ f64         ┆ f64      ┆ f64      ┆ f64      │
# ╞═════════════╪══════════╪══════════╪══════════╡
# │ 30376.03764 ┆ 0.610965 ┆ 0.106377 ┆ 0.282658 │
# └─────────────┴──────────┴──────────┴──────────┘
