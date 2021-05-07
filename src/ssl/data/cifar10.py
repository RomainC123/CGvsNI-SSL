################################################################################
#   Libraries                                                                  #
################################################################################

import os
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms

from PIL import Image

from .image import ImageDatasetContainer
from ..utils.paths import DATASETS_PATH
from ..utils.transforms import IMAGE_TRANSFORMS_TRAIN, IMAGE_TRANSFORMS_TEST

################################################################################
#   CIFAR10 dataset container class                                            #
################################################################################


class CIFAR10DatasetContainer(ImageDatasetContainer):

    def __init__(self, data, nb_samples_total, nb_samples_test, nb_samples_labeled, **kwargs):

        self.epsilon = kwargs['epsilon']

        super(CIFAR10DatasetContainer, self).__init__(data, nb_samples_total, nb_samples_test, nb_samples_labeled, **kwargs)

    def _get_data(self):

        return pd.read_csv(os.path.join(DATASETS_PATH, self.data, 'dataset.csv'))

    def _init_preprocess(self):

        list_imgs = []
        totensor = transforms.ToTensor()

        for img_name in self._df_train_full['Name']:

            img_path = os.path.join(DATASETS_PATH, self.data, 'raw', img_name)
            with open(img_path, 'rb') as f:
                img = Image.open(f)
                if self.img_mode == 'L':
                    img = img.convert('L')
                elif self.img_mode == 'RGB':
                    img = img.convert('RGB')
            list_imgs.append(totensor(img))

        X = torch.stack(list_imgs)
        X_flat = torch.flatten(X, start_dim=1)
        self.means = torch.mean(X_flat, axis=0)
        X_center = X_flat - self.means
        cov = np.cov(X_center, rowvar=False)
        U, S, V = np.linalg.svd(cov)
        self.W = U.dot(np.diag(1.0 / np.sqrt(S + self.epsilon))).dot(U.T)

    def _get_transforms(self):

        return transforms.Compose(IMAGE_TRANSFORMS_TRAIN[self.data][self.img_mode]), transforms.Compose(IMAGE_TRANSFORMS_TEST[self.data][self.img_mode])

    def preprocess(self, input):
        # Input needs to be a (_, 3, 32, 32) tensor with values between 0 and 1
        real_shape = input.shape
        input_flat = torch.flatten(input, start_dim=1)
        input_center = input_flat - self.means
        input_zca = torch.Tensor(self.W.dot(input_center.T).T)
        input_zca_min = torch.min(input_zca, dim=1, keepdim=True)[0]
        input_zca_max = torch.max(input_zca, dim=1, keepdim=True)[0]

        return ((input_zca - input_zca_min) / (input_zca_max - input_zca_min)).reshape(-1, 3, 32, 32)
