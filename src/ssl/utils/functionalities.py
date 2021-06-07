from ..data.mnist import MNISTDatasetContainer
from ..data.cifar10 import CIFAR10DatasetContainer
from ..data.cgvsni import CGvsNIDatasetContainer

from ..models.VGG import VGGContainer
from ..models.CNN import CNNContainer
from ..models.Resnet import Resnet18Container
from ..models.SimpleNet import SimpleNetContainer
from ..models.StandardNet import StandardNetContainer
from ..models.ENet import ENetContainer

from ..optimizers.adam import AdamContainer

from ..methods.tempens import TemporalEnsembling
from ..methods.meanteach import MeanTeacher
from ..methods.only_sup import OnlySup


DATASETS = {
    'CIFAR10': CIFAR10DatasetContainer,
    'MNIST': MNISTDatasetContainer,
    'CGvsNI': CGvsNIDatasetContainer
}

MODELS = {
    'VGG': VGGContainer,
    'CNN': CNNContainer,
    'Resnet18': Resnet18Container,
    'SimpleNet': SimpleNetContainer,
    'StandardNet': StandardNetContainer,
    'ENet': ENetContainer
}

OPTIMIZERS = {
    'Adam': AdamContainer
}

METHODS = {
    'TemporalEnsembling': TemporalEnsembling,
    'MeanTeacher': MeanTeacher,
    'OnlySup': OnlySup
}
