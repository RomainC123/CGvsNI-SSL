import os
import pathlib
import itertools

import datasets
import models

import torchvision.transforms as transforms

TRAIN_STEP = 10  # To be set manually

ROOT_PATH = pathlib.Path(__file__).resolve().parents[1].absolute()

DATASETS_PATH = os.path.join(ROOT_PATH, 'datasets')  # dataset.csv files path
if not os.path.exists(DATASETS_PATH):
    raise RuntimeError('No dataset folder found, please create it')

TRAINED_MODELS_PATH = os.path.join(ROOT_PATH, 'trained_models')  # Graphs path
if not os.path.exists(TRAINED_MODELS_PATH):
    os.makedirs(TRAINED_MODELS_PATH)

DATASETS_IMPLEMENTED = {
    'MNIST': datasets.DatasetMNIST,
    'CIFAR10': datasets.DatasetCIFAR10,
    'CGvsNI': None
}

TRAIN_TRANSFORMS = {
    'MNIST': transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize((0.5), (0.5))]),
    'CIFAR10': transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    'CGvsNI': None
}

TEST_TRANSFORMS = {
    'MNIST': transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize((0.5), (0.5))]),
    'CIFAR10': transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    'CGvsNI': None
}

MODELS = {
    'MNIST': models.MNISTModel(),
    'CIFAR10': models.CIFAR10Model(),
    'CGvsNI': None
}

HYPERPARAMETERS_DEFAULT = {
    'TemporalEnsembling': {
        'alpha': 0.6,
        'ramp_epochs': 10,
        'ramp_mult': 5,
        'max_weight': 30.
    }
}

HYPERPARAMETERS_SEARCH = {
    'TemporalEnsembling': {
        'alpha': [0.6],
        'max_weight': [20., 30., 40.],
        'ramp_epochs': [5, 10, 15],
        'ramp_mult': [2, 5]
    }
}


def get_hyperparameters_combinations(method):

    hyperparameters = HYPERPARAMETERS_SEARCH[method]

    keys, values = zip(*hyperparameters.items())
    permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]

    return permutations_dicts
