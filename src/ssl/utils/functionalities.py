from ..data.image import ImageDatasetContainer
from ..models.VGG import VGGContainer
from ..models.CNN import CNNContainer
from ..optimizers.adam import AdamContainer
from ..methods.tempens import TemporalEnsembling

DATASETS = {
    'CIFAR10': ImageDatasetContainer,
    'MNIST': ImageDatasetContainer
}

MODELS = {
    'VGG': VGGContainer,
    'CNN': CNNContainer
}

OPTIMIZERS = {
    'Adam': AdamContainer
}

METHODS = {
    'TemporalEnsembling': TemporalEnsembling
}
