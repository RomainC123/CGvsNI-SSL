################################################################################
#   Libraries                                                                  #
################################################################################

import os
import numpy as np
import theano as th
import theano.tensor as T
import pandas as pd
import torch
import torchvision.transforms as transforms

from PIL import Image
from scipy import linalg

from .image import ImageDatasetContainer
from ..utils.paths import DATASETS_PATH
from ..utils.transforms import IMAGE_TRANSFORMS_TRAIN, IMAGE_TRANSFORMS_TEST

################################################################################
#   CIFAR10 dataset container class                                            #
################################################################################


class CIFAR10DatasetContainer(ImageDatasetContainer):

    def __init__(self, data, nb_samples_total, nb_samples_test, nb_samples_labeled, cuda_state, **kwargs):

        self.epsilon = kwargs['epsilon']
        self.cuda_state = cuda_state

        super(CIFAR10DatasetContainer, self).__init__(data, nb_samples_total, nb_samples_test, nb_samples_labeled, **kwargs)

    def _get_data(self):

        return pd.read_csv(os.path.join(DATASETS_PATH, self.data, 'dataset.csv'))

    def _init_preprocess(self):

        print('Building ZCA transform...')

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
            list_imgs.append(np.asarray(img))

        x = np.array(list_imgs)
        regularization = self.epsilon

        s = x.shape
        x = x.copy().reshape((s[0], np.prod(s[1:])))
        m = np.mean(x, axis=0)
        x = x - m
        sigma = np.dot(x.T, x) / x.shape[0]
        U, S, V = linalg.svd(sigma)
        tmp = np.dot(U, np.diag(1. / np.sqrt(S + regularization)))
        tmp2 = np.dot(U, np.diag(np.sqrt(S + regularization)))
        self.W = torch.Tensor(np.dot(tmp, U.T))
        self.means = torch.Tensor(m)

        if self.cuda_state:
            self.W = self.W.cuda()
            self.means = self.means.cuda()

    def _get_transforms(self):

        return IMAGE_TRANSFORMS_TRAIN[self.data][self.img_mode], IMAGE_TRANSFORMS_TEST[self.data][self.img_mode]

    def preprocess(self, input):
        # Input needs to be a (_, 3, 32, 32) tensor with values between 0 and 1
        real_shape = input.shape
        input_flat = torch.flatten(input, start_dim=1)
        input_center = input_flat - self.means
        input_zca = torch.transpose(torch.matmul(self.W, torch.transpose(input_center, 0, 1)), 0, 1)
        input_zca_min = torch.min(input_zca, dim=1, keepdim=True)[0]
        input_zca_max = torch.max(input_zca, dim=1, keepdim=True)[0]

        return ((input_zca - input_zca_min) / (input_zca_max - input_zca_min)).reshape(-1, 3, 32, 32)
