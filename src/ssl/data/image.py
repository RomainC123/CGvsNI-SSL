################################################################################
#   Libraries                                                                  #
################################################################################

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from PIL import Image
from torch.utils.data import Dataset

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
        if 'label_transform' in kwargs.keys():
            self.label_transform = kwargs['label_transform']

    def _loader(self, idx, img_mode):

        img_path = os.path.join(self._raw_data_path, self.df_data['Name'][idx])

        with open(img_path, 'rb') as f:
            img = Image.open(f)
            if self.img_mode == 'L':
                return img.convert('L')  # convert image to grey
            elif self.img_mode == 'RGB':
                return img.convert('RGB')  # convert image to rgb image

    def show_img(self, idx):
        # TODO
        pass

################################################################################
#   Image container class                                                      #
################################################################################


class ImageDatasetContainer(BaseDatasetContainer):

    def __init__(self, data, nb_samples_test, nb_samples_labeled):
        super(ImageDatasetContainer, self).__init__(data, nb_samples_test, nb_samples_labeled)

    def make_dataloaders(self, dataloader_params, **kwargs):

        img_mode = kwargs['img_mode']

        self._dataset_train = ImageDataset(self.data,
                                           self._df_train_masked,
                                           img_mode=img_mode,
                                           transform=IMAGE_TRANSFORMS_TRAIN[img_mode])
        self._dataset_valuation = ImageDataset(self.data,
                                               self._df_valuation,
                                               img_mode=img_mode,
                                               transform=IMAGE_TRANSFORMS_TRAIN[img_mode])
        self._dataset_test = ImageDataset(self.data,
                                          self._df_test,
                                          img_mode=img_mode,
                                          transform=IMAGE_TRANSFORMS_TEST[img_mode])

        super(ImageDatasetContainer, self).make_dataloaders(dataloader_params, **kwargs)
