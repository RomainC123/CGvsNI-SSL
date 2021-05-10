import os
import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from distutils.dir_util import copy_tree
from datetime import datetime
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt

from ssl.utils.constants import BATCH_SIZE, DEFAULT_EPOCHS
from ssl.utils.paths import TRAINED_MODELS_PATH
from ssl.utils.functionalities import DATASETS, MODELS, OPTIMIZERS, METHODS
from ssl.utils.hyperparameters import METHODS_DEFAULT, OPTIMIZERS_DEFAULT
from ssl.utils.paths import DATASETS_PATH

data = 'CIFAR10'
nb_samples_total = 60000
nb_samples_test = 10000
nb_samples_labeled = 1000
img_mode = 'RGB'

np.random.seed(0)

dataset = DATASETS[data](data, nb_samples_total, nb_samples_test, nb_samples_labeled, True, img_mode=img_mode, epsilon=1e-1)
train_dataloader, _, _ = dataset.get_dataloaders(True)


def plotImage(X_init, X_processed):
    plt.figure(figsize=(4, 8))
    plt.subplot(211)
    plt.imshow(X_init.cpu().permute(1, 2, 0))
    plt.subplot(212)
    plt.imshow(X_processed.cpu().permute(1, 2, 0))
    plt.show()


pbar = enumerate(train_dataloader)
for batch_idx, (data, target) in pbar:

    data = data.cuda()
    target = target.cuda()

    data_preprocessed = dataset.preprocess(data)
    print(data_preprocessed)
    plotImage(data[0], data_preprocessed[0])
    plotImage(data[1], data_preprocessed[1])
    plotImage(data[2], data_preprocessed[2])
    plotImage(data[3], data_preprocessed[3])
    plotImage(data[4], data_preprocessed[4])

    print(bleh)
