from ssl.models.VGG import VGGContainer
from ssl.optimizers.adam import AdamOptimizer
import os
from ssl.utils.paths import TRAINED_MODELS_PATH
from ssl.data.image import ImageDatasetContainer
from ssl.methods.tempens import TemporalEnsembling
from ssl.utils.hyperparameters import HYPERPARAMETERS_DEFAULT

dataset = ImageDatasetContainer('CIFAR10', 10000, 1000)
model = VGGContainer('normal')
optimizer = AdamOptimizer(max_lr=0.001, beta1=0.9, beta2=0.999)
method = TemporalEnsembling(HYPERPARAMETERS_DEFAULT['TemporalEnsembling'])

model.cuda()
method.cuda()

method.train(dataset, model, optimizer, 0, 2, TRAINED_MODELS_PATH, True, img_mode='RGB')
