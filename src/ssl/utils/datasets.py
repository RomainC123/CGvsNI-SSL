from ..data.mnist import MNISTDatasetContainer
from ..data.cifar10 import CIFAR10DatasetContainer
from ..data.cgvsni import CGvsNIDatasetContainer

DATASETS = {
    'CIFAR10': CIFAR10DatasetContainer,
    'MNIST': MNISTDatasetContainer,
    'CGvsNI': CGvsNIDatasetContainer
}
