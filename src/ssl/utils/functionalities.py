from ..data.image import ImageDatasetContainer
from ..data.cgvsni import CGvsNIDatasetContainer

from ..models.VGG import VGGContainer
from ..models.CNN import CNNContainer
from ..models.Resnet import Resnet18Container
from ..models.SimpleNet import SimpleNetContainer
from ..models.ENet import ENetContainer

from ..optimizers.adam import AdamContainer

from ..methods.tempens import TemporalEnsembling
from ..methods.tempens_new_loss import TemporalEnsemblingNewLoss


DATASETS = {
    'CIFAR10': ImageDatasetContainer,
    'MNIST': ImageDatasetContainer,
    'CGvsNI': CGvsNIDatasetContainer
}

MODELS = {
    'VGG': VGGContainer,
    'CNN': CNNContainer,
    'Resnet18': Resnet18Container,
    'SimpleNet': SimpleNetContainer,
    'ENet': ENetContainer
}

OPTIMIZERS = {
    'Adam': AdamContainer
}

METHODS = {
    'TemporalEnsembling': TemporalEnsembling,
    'TemporalEnsemblingNewLoss': TemporalEnsemblingNewLoss
}
