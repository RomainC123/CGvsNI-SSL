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

from .base import BaseDatasetContainer, BaseDataset
from ..utils.paths import DATASETS_PATH

################################################################################
#   Image dataset class                                                        #
################################################################################


class ImageDataset(BaseDataset):

    def __init__(self, data, df_data, **kwargs):

        super(ImageDataset, self).__init__(data, df_data, **kwargs)

        self._raw_data_path = os.path.join(DATASETS_PATH, data, 'raw')
        if not os.path.exists(self._raw_data_path):
            raise RuntimeError(f'Data type not implemented: {self.data}')

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

        target = self.df_data['Label'][idx]

        if self.transform is not None:
            img = self.transform(img)
        if self.label_transform is not None:
            target = self.label_transform(target)

        return img, target

    def show_img(self, idx):

        img, _ = self._loader(idx)

        plt.figure()
        plt.imshow(img.permute(1, 2, 0))
        plt.show()

################################################################################
#   Image container class                                                      #
################################################################################


class ImageDatasetContainer(BaseDatasetContainer):

    def __init__(self, data, nb_samples_total, nb_samples_test, nb_samples_labeled, **kwargs):

        self.img_mode = kwargs['img_mode']

        super(ImageDatasetContainer, self).__init__(data, nb_samples_total, nb_samples_test, nb_samples_labeled)

    def _get_transforms(self):
        # To overload
        pass

    def get_dataloaders(self, cuda_state):

        transforms_train, transforms_test = self._get_transforms()

        self._dataset_train = ImageDataset(self.data,
                                           self._df_train_masked,
                                           img_mode=self.img_mode,
                                           transform=transforms_train)
        self._dataset_valuation = ImageDataset(self.data,
                                               self._df_valuation,
                                               img_mode=self.img_mode,
                                               transform=transforms_train)
        self._dataset_test = ImageDataset(self.data,
                                          self._df_test,
                                          img_mode=self.img_mode,
                                          transform=transforms_test)

        return super(ImageDatasetContainer, self).get_dataloaders(cuda_state)
