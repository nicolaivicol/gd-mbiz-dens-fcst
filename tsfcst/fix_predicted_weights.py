import polars as pl
import os

from tsfcst.predict_weights_rates import load_predicted
from utils import set_display_options, describe_numeric
import config
import manual_labels


set_display_options()

model_names = ['naive', 'ema', 'theta']
weights_alias = 'full-weight-folds_1-active-20221201-active-target-naive_ema_theta-corner-20221201-20220801-manual_fix'
# weights_alias = 'test-weight-folds_5-active-20220701-active-target-naive_ema_theta-corner-20221201-20220801-manual_fix'

df = load_predicted(weights_alias)
print(df.head().to_pandas())

# load all params
stats_before = df.select(model_names).mean().with_columns(pl.lit('before').alias('when'))

# override from manual labels
dict_override = {
    'naive': manual_labels.NAIVE,
    'ema': manual_labels.MA,
    'ma': manual_labels.MA,
    'theta': manual_labels.TREND,
}

for m_overriding in model_names:
    override = [cfips in dict_override[m_overriding] for cfips in df['cfips']]
    df = pl.concat([df, pl.DataFrame({'override': override})], how='horizontal')
    for m_existing in model_names:
        if m_overriding == m_existing:
            df = df.with_columns(pl.when(pl.col('override')).then(1).otherwise(pl.col(m_overriding)).alias(m_overriding))
        else:
            df = df.with_columns(pl.when(pl.col('override')).then(0).otherwise(pl.col(m_existing)).alias(m_existing))
    df = df.drop('override')

stats_after_override = df.select(model_names).mean().with_columns(pl.lit('after_override').alias('when'))

print('summary stats:')
print(describe_numeric(df.to_pandas()))

print('sample:')
print(df.head(100).to_pandas())

print('distribution by model, before and after fix:')
stats_after_all = df.select(model_names).mean().with_columns(pl.lit('after_all').alias('when'))
print(pl.concat([stats_before, stats_after_override, stats_after_all]))

dir_out = f'{config.DIR_ARTIFACTS}/predict_weights_rates-overriden/{weights_alias}'
os.makedirs(dir_out, exist_ok=True)
df.write_csv(f'{dir_out}/predicted.csv', float_precision=4)

# ┌──────────┬──────────┬──────────┬────────────────┐
# │ naive    ┆ ema      ┆ theta    ┆ when           │
# ╞══════════╪══════════╪══════════╪════════════════╡
# │ 0.635465 ┆ 0.076234 ┆ 0.288301 ┆ before         │
# │ 0.605923 ┆ 0.05522  ┆ 0.338856 ┆ after_override │
# │ 0.605923 ┆ 0.05522  ┆ 0.338856 ┆ after_all      │
# └──────────┴──────────┴──────────┴────────────────┘
