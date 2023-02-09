from tsfcst.models.ma import MovingAverageModel
from tsfcst.models.hw import HoltWintersSmModel
from tsfcst.models.prophet import ProphetModel
from tsfcst.models.theta import ThetaSmModel

MODELS = {
    'ma': MovingAverageModel,
    'hw': HoltWintersSmModel,
    'prophet': ProphetModel,
    'theta': ThetaSmModel,
}
