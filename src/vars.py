import os
import pathlib
import itertools

import models

import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, f1_score

TRAIN_STEP = 30  # To be set manually
TEST_RUNS = 3
ONLY_SUP_RUNS = 10

ROOT_PATH = pathlib.Path(__file__).resolve().parents[1].absolute()

DATASETS_PATH = os.path.join(ROOT_PATH, 'datasets')  # dataset.csv files path
if not os.path.exists(DATASETS_PATH):
    raise RuntimeError('No dataset folder found, please create it')

TRAINED_MODELS_PATH = os.path.join(ROOT_PATH, 'trained_models')  # Graphs path
if not os.path.exists(TRAINED_MODELS_PATH):
    os.makedirs(TRAINED_MODELS_PATH)

METRICS = {
    'accuracy': accuracy_score,
}

TRAIN_TRANSFORMS = {
    'L': transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize((0.5), (0.5))]),
    'RGB': transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    'CGvsNI': None
}

TEST_TRANSFORMS = {
    'L': transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize((0.5), (0.5))]),
    'RGB': transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    'CGvsNI': None
}

HYPERPARAMETERS_DEFAULT = {
    'TemporalEnsembling': {
        'alpha': 0.6,
        'ramp_epochs': 80,
        'ramp_mult': 5,
        'unsup_loss_max_weight': 30.
    }
}

HYPERPARAMETERS_SEARCH = {
    'TemporalEnsembling': {
        'alpha': [0.6],
        'ramp_epochs': [5],
        'ramp_mult': [2, 5],
        'unsup_loss_max_weight': [20.]
    }
}
