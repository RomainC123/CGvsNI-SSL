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

################################################################################
#   Image dataset class                                                        #
################################################################################


class ImageDataset(Dataset):
    """
    Dataset parent class containing images and corresponding target (the class index, or no label)
    ------
    Args:
        - data: data type to use for the dataset
        - img_mode: mode to use to open images
        - df_imgs: dataframe containing the images names and their labels
        - transform: transforms to be applied to all imgs
        - target_transform: same thing for all targets
    Methods:
        len and getitem
    """

    def __init__(self, data, img_mode, df_imgs, transform=None, label_transform=None):

        self.data = data
        self.img_mode = img_mode
        self.df_imgs = df_imgs
        self.transform = transform
        self.label_transform = label_transform

        self._raw_data_path = os.path.join(DATASETS_PATH, data, 'raw')
        if not os.path.exists(self.raw_data_path):
            raise RuntimeError(f'Data type not implemented: {self.data}')

    def _img_loader(self, img_path, img_mode):

        with open(img_path, 'rb') as f:
            img = Image.open(f)
            if mode == 'L':
                return img.convert('L')  # convert image to grey
            elif mode == 'RGB':
                return img.convert('RGB')  # convert image to rgb image

    def __len__(self):
        return len(self.df_imgs)

    def __getitem__(self, idx):

        if idx > len(self.df_imgs):
            raise ValueError(f'Index out of bounds: {idx}')

        img_path = os.path.join(self._raw_data_path, self.df_imgs['Name'][idx])
        img = self._img_loader(img_path, self.img_mode)
        label = self.df_imgs['Label'][idx]

        if self.transform is not None:
            img = self.transform(img)
        if self.label_transform is not None:
            label = self.label_transform(label)

        return img, label

    def show_img(self, idx):
        # TODO
        pass
