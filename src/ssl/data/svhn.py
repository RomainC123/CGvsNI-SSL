################################################################################
#   Libraries                                                                  #
################################################################################

import os
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms

from PIL import Image
from scipy import linalg
from torch.utils.data.sampler import RandomSampler

from .image import ImageDatasetContainer
from ..utils.paths import DATASETS_PATH
from ..utils.constants import PERCENT_LABELS_BATCH
from ..utils.transforms import IMAGE_TRANSFORMS_TRAIN, IMAGE_TRANSFORMS_TEST
from ..utils.tools import TwoStreamBatchSampler

################################################################################
#   CIFAR10 dataset container class                                            #
################################################################################


class SVHNDatasetContainer(ImageDatasetContainer):

    def __init__(self, data, nb_samples_total, nb_samples_test, nb_samples_labeled, cuda_state, **kwargs):

        self.cuda_state = cuda_state

        super(SVHNDatasetContainer, self).__init__(data, nb_samples_total, nb_samples_test, nb_samples_labeled, **kwargs)

    def _get_data(self):

        return pd.read_csv(os.path.join(DATASETS_PATH, self.data, 'dataset.csv'))

    def _get_transforms(self):

        return IMAGE_TRANSFORMS_TRAIN[self.data][self.img_mode], IMAGE_TRANSFORMS_TEST[self.data][self.img_mode]

    def _init_preprocess(self):

        print('Building zero mean transform...')

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

        list_imgs = torch.stack(list_imgs)
        self.means = torch.mean(list_imgs, dim=(0, 2, 3))
        self.stds = torch.std(list_imgs, dim=(0, 2, 3))

        if self.cuda_state:
            self.means = self.means.cuda()
            self.stds = self.stds.cuda()

    def preprocess(self, input):

        return (input - self.means[:, None, None]) / self.stds[:, None, None]

    def _set_samplers(self):

        unmasked_idx = self._df_train_masked.loc[self._df_train_masked['Label'] != -1].index
        masked_idx = self._df_train_masked.loc[self._df_train_masked['Label'] == -1].index

        self._batch_sampler_train = TwoStreamBatchSampler(masked_idx, unmasked_idx, self.batch_size, int(self.batch_size * PERCENT_LABELS_BATCH))
