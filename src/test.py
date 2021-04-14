from ssl.models.VGG import VGGContainer
from ssl.optimizers.adam import AdamContainer
import os
from datetime import datetime
from ssl.utils.paths import TRAINED_MODELS_PATH
from ssl.data.image import ImageDatasetContainer
from ssl.methods.tempens import TemporalEnsembling
from ssl.utils.hyperparameters import METHODS_DEFAULT
from ssl.utils.tools import show_graphs, show_schedules

import torch
import numpy as np

SEED = 0

np.random.seed(SEED) # Fixed dataset generator

show_graphs(os.path.join(TRAINED_MODELS_PATH, 'CIFAR10_14-04-2021_15:58:13'))
show_schedules(1, 20)
