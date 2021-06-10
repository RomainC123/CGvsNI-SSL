from ..data.cifar10 import CIFAR10DatasetContainer
from ..data.svhn import SVHNDatasetContainer
from ..data.mnist import MNISTDatasetContainer
from ..data.cgvsni import CGvsNIDatasetContainer

DATASETS = {
    'CIFAR10': CIFAR10DatasetContainer,
    'SVHN': SVHNDatasetContainer,
    'MNIST': MNISTDatasetContainer,
    'CGvsNI': CGvsNIDatasetContainer
}
