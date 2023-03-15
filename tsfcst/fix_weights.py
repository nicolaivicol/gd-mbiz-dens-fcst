import polars as pl
import os

from tsfcst.find_best_params import load_best_params
from tsfcst.find_best_weights import load_best_weights
from utils import set_display_options, describe_numeric
import config
import manual_labels


set_display_options()

# weights_alias = 'active-feats-naive_ema_theta-corner-20220701-no'
# dict_model_params = {
#     'naive': 'active-cfips-20220701-naive-test-tld-1-0_0',
#     'ema': 'active-cfips-20220701-ema-test-tld-20-0_0',
#     'theta': 'active-cfips-20220701-theta-test-tld-50-0_02'
# }

# weights_alias = 'active-target-naive_ema_theta-corner-20221201-20220801'
# dict_model_params = {
#     'naive': 'active-cfips-20220701-naive-test-tld-1-0_0',
#     'ema': 'active-cfips-20220701-ema-test-tld-20-0_0',
#     'theta': 'active-cfips-20220701-theta-test-tld-50-0_02'
# }

weights_alias = 'active-feats-naive_ema_theta-corner-20221201-no'
dict_model_params = {
    'naive': 'active-cfips-20221201-naive-full-tld-1-0_0',
    'ema': 'active-cfips-20221201-ema-full-tld-20-0_0',
    'theta': 'active-cfips-20221201-theta-full-tld-50-0_02'
}

df_weights = load_best_weights(weights_alias, normalize=False)
print(df_weights.head().to_pandas())

# load all params
df = df_weights
stats_before = df.select(dict_model_params.keys()).mean().with_columns(pl.lit('before').alias('when'))

for model, model_params_id in dict_model_params.items():

    df_params = load_best_params(model_params_id)
    df_params = df_params.rename({'smape_cv_opt': f'smape_cv_opt_{model}'})
    cols = ['cfips', f'smape_cv_opt_{model}']

    if model == 'theta':
        df_params = df_params.rename({'theta': 'par_theta'})
        cols.append('par_theta')
    elif model in ['ma', 'ema']:
        df_params = df_params.rename({'window': f'par_window_{model}'})
        cols.append(f'par_window_{model}')

    df = df.join(df_params.select(cols), on='cfips')

print(describe_numeric(df.to_pandas()))

# replace theta with naive where it is equivalent
df = df.with_columns(
    ((pl.col('theta') == 1)
     & (pl.col('par_theta') <= 1.05)
     & ((pl.col('smape_naive') - pl.col('smape_theta')) <= 0.05)
     ).alias('theta_is_naive'))

print(f"{df['theta_is_naive'].sum()/df['theta'].sum()*100:.2f}% of best theta is naive")
print(df.filter(pl.col('theta_is_naive')).head().to_pandas())

# override theta as naive
df = df.with_columns([
    pl.when(pl.col('theta_is_naive')).then(0).otherwise(pl.col('theta')).alias('theta'),
    pl.when(pl.col('theta_is_naive')).then(1).otherwise(pl.col('naive')).alias('naive')
])
df = df.drop('theta_is_naive')
stats_after_theta = df.select(dict_model_params.keys()).mean().with_columns(pl.lit('after_theta').alias('when'))

# replace ma/ema with naive where it is equivalent
ma = 'ema'
df = df.with_columns(
    ((pl.col(ma) == 1)
     & (pl.col(f'par_window_{ma}') <= 1)
     & ((pl.col('smape_naive') - pl.col(f'smape_{ma}')) <= 0.05)
     ).alias(f'{ma}_is_naive'))

n = df[f'{ma}_is_naive'].sum()
print(f"{n/df['theta'].sum()*100:.2f}% of best {ma} is naive")

if n > 0:
    print(df.filter(pl.col(f'{ma}_is_naive')).to_pandas())
    df = df.with_columns([
        pl.when(pl.col(f'{ma}_is_naive')).then(0).otherwise(pl.col(ma)).alias(ma),
        pl.when(pl.col(f'{ma}_is_naive')).then(1).otherwise(pl.col('naive')).alias('naive')
    ])

df = df.drop(f'{ma}_is_naive')
stats_after_ema = df.select(dict_model_params.keys()).mean().with_columns(pl.lit('after_ema').alias('when'))

# override from manual labels
dict_override = {
    'naive': manual_labels.NAIVE,
    'ema': manual_labels.MA,
    'ma': manual_labels.MA,
    'theta': manual_labels.TREND,
}

for m_overriding in dict_model_params.keys():
    override = [cfips in dict_override[m_overriding] for cfips in df['cfips']]
    df = pl.concat([df, pl.DataFrame({'override': override})], how='horizontal')
    for m_existing in dict_model_params.keys():
        if m_overriding == m_existing:
            df = df.with_columns(pl.when(pl.col('override')).then(1).otherwise(pl.col(m_overriding)).alias(m_overriding))
        else:
            df = df.with_columns(pl.when(pl.col('override')).then(0).otherwise(pl.col(m_existing)).alias(m_existing))
    df = df.drop('override')

stats_after_override = df.select(dict_model_params.keys()).mean().with_columns(pl.lit('after_override').alias('when'))

print('summary stats:')
print(describe_numeric(df.to_pandas()))

print('sample:')
print(df.head().to_pandas())

print('distribution by model, before and after fix:')
stats_after_all = df.select(dict_model_params.keys()).mean().with_columns(pl.lit('after_all').alias('when'))
print(pl.concat([stats_before, stats_after_theta, stats_after_ema, stats_after_override, stats_after_all]))

dir_out = f'{config.DIR_ARTIFACTS}/find_best_weights/{weights_alias}-manual_fix'
os.makedirs(dir_out, exist_ok=True)
df.write_csv(f'{dir_out}/{weights_alias}.csv', float_precision=4)

# distribution by model:
#                     naive    ma  theta
# feats-cv-learn:      0.40  0.16   0.44
# target-test-learn:   0.62  0.08   0.30
# feats-cv-pred:       0.45  0.15   0.40
