from ssl.models.VGG import VGGContainer
from ssl.optimizers.adam import AdamOptimizer
import os
from datetime import datetime
from ssl.utils.paths import TRAINED_MODELS_PATH
from ssl.data.image import ImageDatasetContainer
from ssl.methods.tempens import TemporalEnsembling
from ssl.utils.hyperparameters import HYPERPARAMETERS_DEFAULT

import torch
import numpy as np

SEED = 0

np.random.seed(SEED) # Fixed dataset generator

test_path = os.path.join(TRAINED_MODELS_PATH, 'test_{date:%Y-%m-%d_%H:%M:%S}'.format(date=datetime.now()))

dataset = ImageDatasetContainer('CIFAR10', 10000, 1000, img_mode='RGB')
model = VGGContainer('normal')
optimizer = AdamOptimizer(max_lr=0.001, beta1=0.9, beta2=0.999)
method = TemporalEnsembling(HYPERPARAMETERS_DEFAULT['TemporalEnsembling'])

model.cuda()
method.cuda()

method.train(dataset, model, optimizer, 0, 1, test_path, True)
method.test(dataset, model, test_path, False)
