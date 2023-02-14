from typing import Dict

from tsfcst.models.abstract_model import TsModel
from tsfcst.models.inventory import MODELS
from tsfcst.forecasters.forecaster import Forecaster
import polars as pl


class ForecasterConfig:

    def __init__(self, model_cls: type(TsModel), params_model: Dict, params_forecaster: Dict):
        self.model_cls = model_cls
        self.params_model = params_model
        self.params_forecaster = params_forecaster

    @classmethod
    def from_best_params(cls, best_params: Dict) -> 'ForecasterConfig':
        model_name = best_params['model']
        model_cls = MODELS[model_name]
        params_forecaster = {p: best_params[p] for p in Forecaster.names_params() if p in best_params.keys()}
        params_model = {p: best_params[p] for p in model_cls.names_params() if p in best_params.keys()}
        return cls(model_cls=model_cls, params_model=params_model, params_forecaster=params_forecaster)

    @classmethod
    def from_df_with_best_params(cls, cfips: int, df_best_params: pl.DataFrame) -> 'ForecasterConfig':
        best_params = df_best_params.filter(pl.col('cfips') == cfips)
        if len(best_params) > 0 and 'smape_cv_opt' in best_params.columns:
            best_params = best_params.sort('smape_cv_opt')
        best_params = best_params.to_dicts()[0]
        return cls.from_best_params(best_params)