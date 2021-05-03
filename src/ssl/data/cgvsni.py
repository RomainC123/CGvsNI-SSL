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
from sklearn.model_selection import train_test_split

from .image import ImageDataset, ImageDatasetContainer
from ..utils.constants import CGVSNI_DATASETS_IDS, VAL_NUMBER_CGVSNI
from ..utils.paths import DATASETS_PATH

################################################################################
#   CGvsNI dataset class                                                       #
################################################################################


class CGvsNIDatasetContainer(ImageDatasetContainer):

    def __init__(self, data, nb_samples_total, nb_samples_test, nb_samples_labeled, **kwargs):

        self.datasets_to_use = kwargs['datasets_to_use']
        self.label_mode = kwargs['label_mode']

        super(CGvsNIDatasetContainer, self).__init__(data, nb_samples_total, nb_samples_test, nb_samples_labeled, **kwargs)

    def _split_data(self):

        ids_to_keep = [0]
        for dataset in self.datasets_to_use.split('_'):
            ids_to_keep.append(CGVSNI_DATASETS_IDS[dataset])
        df_data = pd.read_csv(os.path.join(DATASETS_PATH, self.data, 'dataset.csv'))
        df_data = df_data.loc[df_data['Label'].isin(ids_to_keep)]

        if self.label_mode == 'Biclass':
            df_data['Label'] = df_data['Label'].apply(lambda x: 1 if x > 0 else 0)

        if self.nb_samples_total != -1:
            df_data, _ = train_test_split(df_data, train_size=self.nb_samples_total, shuffle=True, stratify=df_data['Label'])
        df_train, df_test = train_test_split(df_data, test_size=self.nb_samples_test, shuffle=True, stratify=df_data['Label'])
        _, df_valuation = train_test_split(df_train, test_size=VAL_NUMBER_CGVSNI, shuffle=True, stratify=df_train['Label'])

        self._df_train_full = df_train.reset_index(drop=True)
        self._df_valuation = df_valuation.reset_index(drop=True)
        self._df_test = df_test.reset_index(drop=True)
