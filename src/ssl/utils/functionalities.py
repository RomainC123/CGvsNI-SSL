from ..data.image import ImageDatasetContainer
from ..models.VGG import VGGContainer
from ..optimizers.adam import AdamContainer
from ..methods.tempens import TemporalEnsembling

DATASETS = {
    'CIFAR10': ImageDatasetContainer
}

MODELS = {
    'VGG': VGGContainer
}

OPTIMIZERS = {
    'Adam': AdamContainer
}

METHODS = {
    'TemporalEnsembling': TemporalEnsembling
}
