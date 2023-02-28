from tsfcst.models.ma import MovingAverageModel, ExponentialMovingAverageModel, SimpleMovingAverageModel
from tsfcst.models.hw import HoltWintersSmModel
from tsfcst.models.prophet import ProphetModel
from tsfcst.models.theta import ThetaSmModel
from tsfcst.models.naive import NaiveModel
from tsfcst.models.arima import ArimaModel
from tsfcst.models.ets import ETSsmModel
from tsfcst.models.drift import DriftModel

MODELS = {
    'naive': NaiveModel,
    'ma': MovingAverageModel,
    'ema': ExponentialMovingAverageModel,
    'sma': SimpleMovingAverageModel,
    'drift': DriftModel,
    'theta': ThetaSmModel,
    'ets': ETSsmModel,
    'hw': HoltWintersSmModel,
    'prophet': ProphetModel,
    'arima': ArimaModel,
}
