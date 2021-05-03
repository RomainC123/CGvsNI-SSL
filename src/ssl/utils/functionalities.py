from ..data.image import ImageDatasetContainer

from ..models.VGG import VGGContainer
from ..models.CNN import CNNContainer
from ..models.Resnet import Resnet18Container
from ..models.SimpleNet import SimpleNetContainer

from ..optimizers.adam import AdamContainer

from ..methods.tempens import TemporalEnsembling
from ..methods.tempens_new_loss import TemporalEnsemblingNewLoss


DATASETS = {
    'CIFAR10': ImageDatasetContainer,
    'MNIST': ImageDatasetContainer
}

MODELS = {
    'VGG': VGGContainer,
    'CNN': CNNContainer,
    'Resnet18': Resnet18Container,
    'SimpleNet': SimpleNetContainer
}

OPTIMIZERS = {
    'Adam': AdamContainer
}

METHODS = {
    'TemporalEnsembling': TemporalEnsembling,
    'TemporalEnsemblingNewLoss': TemporalEnsemblingNewLoss
}
