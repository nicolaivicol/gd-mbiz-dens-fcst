from tsfcst.models.ma import MovingAverageModel
from tsfcst.models.hw_sm import HoltWintersSmModel
from tsfcst.models.prophet import ProphetModel

MODELS = {
    'ma': MovingAverageModel,
    'hw_sm': HoltWintersSmModel,
    'prophet': ProphetModel,
}
