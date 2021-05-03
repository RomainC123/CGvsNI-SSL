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

from .image import ImageDataset, ImageDatasetContainer
from ..utils.constants import CGVSNI_DATASETS_IDS
from ..utils.paths import DATASETS_PATH

################################################################################
#   CGvsNI dataset class                                                       #
################################################################################


class CGvsNIDatasetContainer(ImageDatasetContainer):

    def __init__(self, data, df_data, **kwargs):

        self.label_mode = kwargs['label_mode']
        self.datasets_to_use = kwargs['datasets_to_use']

        super(CGvsNIDataset, self).__init__(data, df_data, **kwargs)

    def _split_data(self):

        ids_to_keep = []
        for dataset in self.datasets_to_use:
            ids_to_keep.append(CGVSNI_DATASETS_IDS[dataset])
        df_data = pd.read_csv(os.path.join(DATASETS_PATH, self.data, 'dataset.csv')).loc[df_data['Label'] in ids_to_keep]

        if self.label_mode == 'Biclass':
            df_data['Label'] = df_data['Label'].apply(lambda x: 1 if x > 0 else 0)

        if self.nb_samples_total != -1:
            df_data, _ = train_test_split(df_data, train_size=self.nb_samples_total, shuffle=True, stratify=df_data['Label'])
        df_train, df_test = train_test_split(df_data, test_size=self.nb_samples_test, shuffle=True, stratify=df_data['Label'])
        _, df_valuation = train_test_split(df_train, test_size=VAL_NUMBER, shuffle=True, stratify=df_train['Label'])

        self._df_train_full = df_train.reset_index(drop=True)
        self._df_valuation = df_valuation.reset_index(drop=True)
        self._df_test = df_test.reset_index(drop=True)
