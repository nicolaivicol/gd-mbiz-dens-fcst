import polars as pl
import glob

import config

my_sub = 'model-driftr-lgbm_bestpub_comb_v4_adj2pub_damp-driftr-20221201-active-single-driftr'
public_sub = 'best_public'

dir_fcsts = f'{config.DIR_ARTIFACTS}/forecast_ensemble/{my_sub}'
file_fcst = sorted(glob.glob(f'{dir_fcsts}/sub*.csv'))[-1]
df_my = pl.read_csv(file_fcst)
df_my = df_my \
    .with_columns(pl.col('row_id').str.split('_').alias('row_id_split')) \
    .with_columns(pl.col('row_id_split').arr.get(0).cast(pl.Int64).alias('cfips')) \
    .with_columns(pl.col('row_id_split').arr.get(1).str.strptime(pl.Date, fmt='%Y-%m-%d').alias('date'))

dir_fcsts = f'{config.DIR_ARTIFACTS}/forecast_ensemble/{public_sub}'
file_fcst = sorted(glob.glob(f'{dir_fcsts}/sub*.csv'))[-1]
df_pub = pl.read_csv(file_fcst)
df_pub = df_pub \
    .with_columns(pl.col('row_id').str.split('_').alias('row_id_split')) \
    .with_columns(pl.col('row_id_split').arr.get(0).cast(pl.Int64).alias('cfips')) \
    .with_columns(pl.col('row_id_split').arr.get(1).str.strptime(pl.Date, fmt='%Y-%m-%d').alias('date'))\
    .filter(pl.col('date') == pl.lit('2023-01-01').str.strptime(pl.Date, fmt='%Y-%m-%d'))

df_my = df_my.join(df_pub.select(['row_id', 'microbusiness_density']).rename({'microbusiness_density': 'new'}),
                   on='row_id', how='left')

df_my = df_my.with_columns(pl.when(pl.col('new').is_null()).then(pl.col('microbusiness_density')).otherwise(pl.col('new')).alias('microbusiness_density'))

df_my = df_my \
    .sort(['cfips', 'date']) \
    .select(['row_id', 'microbusiness_density']) \
    .write_csv(f'{config.DIR_ARTIFACTS}/forecast_ensemble/combined/sub-{my_sub}.csv', float_precision=6)

print('')
