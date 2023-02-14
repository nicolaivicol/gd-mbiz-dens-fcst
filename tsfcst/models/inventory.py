from tsfcst.models.ma import MovingAverageModel
from tsfcst.models.hw import HoltWintersSmModel
from tsfcst.models.prophet import ProphetModel
from tsfcst.models.theta import ThetaSmModel
from tsfcst.models.naive import NaiveModel

MODELS = {
    'naive': NaiveModel,
    'ma': MovingAverageModel,
    'theta': ThetaSmModel,
    'hw': HoltWintersSmModel,
    'prophet': ProphetModel,
}
