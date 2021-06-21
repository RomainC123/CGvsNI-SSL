################################################################################
#   Libraries                                                                  #
################################################################################

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data.sampler import RandomSampler, BatchSampler

from .image import ImageDataset, ImageDatasetContainer
from ..utils.constants import CGVSNI_DATASETS_IDS, CG_IMG_MULT, VAL_NUMBER, DATA_NO_LABEL
from ..utils.paths import DATASETS_PATH
from ..utils.transforms import IMAGE_TRANSFORMS_TRAIN, IMAGE_TRANSFORMS_TEST

################################################################################
#   CGvsNI dataset container class                                             #
################################################################################


class CGvsNIDatasetContainer(ImageDatasetContainer):

    def __init__(self, data, nb_samples_total, nb_samples_test, nb_samples_labeled, cuda_state, **kwargs):

        self.datasets_to_use = kwargs['datasets_to_use']
        self.label_mode = kwargs['label_mode']

        super(CGvsNIDatasetContainer, self).__init__(data, nb_samples_total, nb_samples_test, nb_samples_labeled, **kwargs)

        self._relabel_data()

    def _get_data(self):

        self.kept_datasets = []

        ids_to_keep = [0]
        for dataset in self.datasets_to_use.split('_'):
            self.kept_datasets.append(CGVSNI_DATASETS_IDS[dataset])
            ids_to_keep.append(CGVSNI_DATASETS_IDS[dataset])
        df_data = pd.read_csv(os.path.join(DATASETS_PATH, self.data, 'dataset.csv'))
        df_data = df_data.loc[df_data['Label'].isin(ids_to_keep)]

        return df_data

    def _relabel_data(self):

        if self.label_mode == 'Biclass':
            self._df_train_labeled['Label'] = self._df_train_labeled['Label'].apply(lambda x: 1 if x > 0 else 0)
            self._df_train_masked['Label'] = self._df_train_masked['Label'].apply(lambda x: x if x == -1 else 1 if x > 0 else 0)
            self._df_valuation['Label'] = self._df_valuation['Label'].apply(lambda x: 1 if x > 0 else 0)
            self._df_test['Label'] = self._df_test['Label'].apply(lambda x: 1 if x > 0 else 0)

    def _split_data(self):
        """
        Creates self._df_train_full, self._df_test and self._df_val
        """

        df_data = self._get_data()

        nb_ni_train = (self.nb_samples_total - self.nb_samples_test) / 2
        nb_cg_train = nb_ni_train / CG_IMG_MULT
        assert nb_ni_train.is_integer()
        assert nb_cg_train.is_integer()

        nb_ni_val = VAL_NUMBER[self.data] / 2
        nb_cg_val = nb_ni_val
        assert nb_ni_val.is_integer()
        assert nb_cg_val.is_integer()

        nb_ni_test = self.nb_samples_test / 2
        nb_cg_test = nb_ni_test
        assert nb_ni_test.is_integer()
        assert nb_cg_test.is_integer()

        df_ni = df_data.loc[df_data['Label'] == 0]
        df_cg = df_data.loc[df_data['Label'].isin(self.kept_datasets)]

        df_ni_train, rest_df_ni = train_test_split(df_ni, train_size=int(nb_ni_train))
        df_cg_train_no_mult, rest_df_cg = train_test_split(df_cg, train_size=int(nb_cg_train), stratify=df_cg['Label'])

        df_ni_val = df_ni_train.sample(n=int(nb_ni_val))
        df_cg_val = df_cg_train_no_mult.sample(n=int(nb_cg_val))

        df_ni_test = rest_df_ni.sample(n=int(nb_ni_test))
        df_cg_test = rest_df_ni.sample(n=int(nb_cg_test))

        df_train = pd.concat([df_ni_train, df_cg_train_no_mult])
        df_valuation = pd.concat([df_ni_val, df_cg_val])
        df_test = pd.concat([df_ni_test, df_cg_test])

        self._df_train_full = df_train.reset_index(drop=True)
        self._df_valuation = df_valuation.reset_index(drop=True)
        self._df_test = df_test.reset_index(drop=True)

    def _mask_data(self):
        """
        Creates self._df_train_masked (frame with all train images and masked and unmasked labels) and self._df_train_labeled (only rows that stayed labeled)
        """

        nb_labels = self.nb_samples_labeled * (1 + CG_IMG_MULT) / (2 * CG_IMG_MULT)
        assert nb_labels.is_integer()

        if self.nb_samples_labeled != -1 and self.nb_samples_labeled != len(self._df_train_full):
            df_masked, df_labeled = train_test_split(self._df_train_full, test_size=int(nb_labels), shuffle=True, stratify=self._df_train_full['Label'])
            self._df_train_labeled = df_labeled.reset_index(drop=True)
        else:
            df_masked = pd.DataFrame()

        self._df_train_masked = self._df_train_full.copy()
        df_ni_full = self._df_train_full.loc[self._df_train_full['Label'] == 0]
        df_cg_full_no_mult = self._df_train_full.loc[self._df_train_full['Label'] != 0]
        self._df_train_full = pd.concat([df_ni_full] + [df_cg_full_no_mult] * CG_IMG_MULT, ignore_index=True)

        cg_idx = self._df_train_masked.loc[self._df_train_masked['Label'] != 0].index

        self._df_train_masked.loc[df_masked.index, 'Label'] = DATA_NO_LABEL
        df_train_cg_masked = self._df_train_masked.iloc[cg_idx]
        self._df_train_masked.drop(cg_idx, inplace=True)
        self._df_train_masked = pd.concat([self._df_train_masked] + [df_train_cg_masked] * CG_IMG_MULT, ignore_index=True)
        self._df_train_masked = self._df_train_masked.reset_index(drop=True)

    def _get_transforms(self):

        return IMAGE_TRANSFORMS_TRAIN[self.data][self.img_mode], IMAGE_TRANSFORMS_TEST[self.data][self.img_mode]

    def _set_samplers(self):

        sampler_train = RandomSampler(self._dataset_train)
        self._batch_sampler_train = BatchSampler(sampler_train, self.batch_size, drop_last=True)
