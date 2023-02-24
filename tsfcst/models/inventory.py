from tsfcst.models.ma import MovingAverageModel
from tsfcst.models.hw import HoltWintersSmModel
from tsfcst.models.prophet import ProphetModel
from tsfcst.models.theta import ThetaSmModel
from tsfcst.models.naive import NaiveModel
from tsfcst.models.arima import ArimaModel
from tsfcst.models.ets import ETSsmModel

MODELS = {
    'naive': NaiveModel,
    'ma': MovingAverageModel,
    'theta': ThetaSmModel,
    'ets': ETSsmModel,
    'hw': HoltWintersSmModel,
    'prophet': ProphetModel,
    'arima': ArimaModel,
}
