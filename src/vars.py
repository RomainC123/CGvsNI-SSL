import os
import pathlib

import datasets
import models

import torchvision.transforms as transforms

TRAIN_STEP = 10  # To be set manually

HYPERPARAMETERS = {
    'TemporalEnsembling': {
        'alpha': 0.6,
        'ramp_epochs': 80,
        'ramp_mult': 5,
        'max_weight': 30.
    }
}

DATASETS_IMPLEMENTED = {
    'MNIST': datasets.DatasetMNIST,
    'CIFAR10': datasets.DatasetCIFAR10,
    'CGvsNI': None
}

TRANSFORMS_TRAIN = {
    'MNIST': transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize((0.5), (0.5))]),
    'CIFAR10': transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    'CGvsNI': None
}

TRANSFORMS_TEST = {
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

ROOT_PATH = pathlib.Path(__file__).resolve().parents[1].absolute()

DATASETS_PATH = os.path.join(ROOT_PATH, 'datasets')  # dataset.csv files path
if not os.path.exists(DATASETS_PATH):
    os.makedirs(DATASETS_PATH)

GRAPHS_PATH = os.path.join(ROOT_PATH, 'graphs')  # Graphs path
if not os.path.exists(GRAPHS_PATH):
    os.makedirs(GRAPHS_PATH)

LOGS_PATH = os.path.join(ROOT_PATH, 'logs')  # Logs path
if not os.path.exists(LOGS_PATH):
    os.makedirs(LOGS_PATH)

TEST_RESULTS_PATH = os.path.join(ROOT_PATH, 'test_results')
if not os.path.exists(TEST_RESULTS_PATH):
    os.makedirs(TEST_RESULTS_PATH)
