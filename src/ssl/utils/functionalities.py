from ..data.image import ImageDatasetContainer
from ..models.VGG import VGGContainer
from ..models.MNIST import MNISTModelContainer
from ..optimizers.adam import AdamContainer
from ..methods.tempens import TemporalEnsembling

DATASETS = {
    'CIFAR10': ImageDatasetContainer,
    'MNIST': ImageDatasetContainer
}

MODELS = {
    'VGG': VGGContainer,
    'MNISTModel': MNISTModelContainer
}

OPTIMIZERS = {
    'Adam': AdamContainer
}

METHODS = {
    'TemporalEnsembling': TemporalEnsembling
}
