################################################################################
#   Libraries                                                                  #
################################################################################

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from PIL import Image
from torch.utils.data import Dataset, DataLoader

from ..utils.paths import DATASETS_PATH
from .base import BaseDatasetContainer, BaseDataset
from ..utils.transforms import IMAGE_TRANSFORMS_TRAIN, IMAGE_TRANSFORMS_TEST

################################################################################
#   Image dataset class                                                        #
################################################################################


class ImageDataset(BaseDataset):

    def __init__(self, data, df_data, **kwargs):

        super(ImageDataset, self).__init__(data, df_data, **kwargs)

        self.img_mode = kwargs['img_mode']
        if 'transform' in kwargs.keys():
            self.transform = kwargs['transform']
        else:
            self.transform = None
        if 'label_transform' in kwargs.keys():
            self.label_transform = kwargs['label_transform']
        else:
            self.label_transform = None

    def _loader(self, idx):

        img_path = os.path.join(self._raw_data_path, self.df_data['Name'][idx])

        with open(img_path, 'rb') as f:
            img = Image.open(f)
            if self.img_mode == 'L':
                img = img.convert('L')  # convert image to grey
            elif self.img_mode == 'RGB':
                img = img.convert('RGB')  # convert image to rgb image

        if self.transform is not None:
            img = self.transform(img)
        if self.label_transform is not None:
            target = self.label_transform(target)

        return img

    def show_img(self, idx):
        # TODO
        pass

################################################################################
#   Image container class                                                      #
################################################################################


class ImageDatasetContainer(BaseDatasetContainer):

    def __init__(self, data, nb_samples_test, nb_samples_labeled, **kwargs):

        self.img_mode = kwargs['img_mode']

        super(ImageDatasetContainer, self).__init__(data, nb_samples_test, nb_samples_labeled)

    def get_dataloaders_training(self, cuda_state):

        self._dataset_train = ImageDataset(self.data,
                                           self._df_train_masked,
                                           img_mode=self.img_mode,
                                           transform=IMAGE_TRANSFORMS_TRAIN[self.img_mode])
        self._dataset_valuation = ImageDataset(self.data,
                                               self._df_valuation,
                                               img_mode=self.img_mode,
                                               transform=IMAGE_TRANSFORMS_TRAIN[self.img_mode])

        return super(ImageDatasetContainer, self).get_dataloaders_training(cuda_state)

    def get_dataloaders_testing(self, cuda_state):

        self._dataset_test = ImageDataset(self.data,
                                          self._df_test,
                                          img_mode=self.img_mode,
                                          transform=IMAGE_TRANSFORMS_TEST[self.img_mode])

        return super(ImageDatasetContainer, self).get_dataloaders_testing(cuda_state)
