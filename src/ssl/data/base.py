################################################################################
#   Libraries                                                                  #
################################################################################

import os
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

from ..utils.paths import DATASETS_PATH
from ..utils.constants import VAL_NUMBER, DATA_NO_LABEL, DATALOADER_PARAMS_CUDA, DATALOADER_PARAMS_NO_CUDA

################################################################################
#   Dataset class                                                              #
################################################################################


class BaseDataset(Dataset):
    """
    Base class for dataset objects
    -----------------------------------------------
    Overloading:
        In _loader, need to return the train data for the given idx
    """

    def __init__(self, data, df_data, **kwargs):

        self.data = data
        self.df_data = df_data

        self._raw_data_path = os.path.join(DATASETS_PATH, data, 'raw')
        if not os.path.exists(self._raw_data_path):
            raise RuntimeError(f'Data type not implemented: {self.data}')

    def _loader(self, idx):
        # TO OVERLOAD
        pass

    def __len__(self):
        return len(self.df_data)

    def __getitem__(self, idx):

        if idx > len(self.df_data):
            raise ValueError(f'Index out of bounds: {idx}')

        item = self._loader(idx)
        label = self.df_data['Label'][idx]

        return item, label

################################################################################
#   Dataset container class                                                              #
################################################################################


class BaseDatasetContainer:
    """
    Base class for dataset containers
    Given the data to use and the test and labeled splits, creates the frames to then be able to create dataloaders
    -----------------------------------------------
    Methods:
    - make_dataloaders
    - get_dataloaders
    - get_infos
    Overloading:
        In make_dataloaders, the dataset attributes need to be created
    """

    def __init__(self, data, nb_samples_test, nb_samples_labeled):

        self.data = data
        self.nb_samples_test = nb_samples_test
        self.nb_samples_labeled = nb_samples_labeled

        self._split_data()
        self._mask_data()

        self.nb_samples = len(self._df_train_full) + len(self._df_test)
        self.nb_samples_train = len(self._df_train_full)
        self.percent_test = self.nb_samples_test / self.nb_samples
        self.percent_labeled = self.nb_samples_labeled / self.nb_samples_train

        self.nb_classes = len(self._df_train_full['Label'].unique())

    def _split_data(self):
        """
        Creates self._df_train_full, self._df_test and self._df_val
        """

        df_data = pd.read_csv(os.path.join(DATASETS_PATH, self.data, 'dataset.csv'))

        df_train, df_test = train_test_split(df_data, test_size=self.nb_samples_test, shuffle=True, stratify=df_data['Label'])
        _, df_valuation = train_test_split(df_train, test_size=VAL_NUMBER, shuffle=True, stratify=df_train['Label'])

        self._df_train_full = df_train.reset_index(drop=True)
        self._df_valuation = df_valuation.reset_index(drop=True)
        self._df_test = df_test.reset_index(drop=True)

    def _mask_data(self):
        """
        Creates self._df_train_masked (frame with all train images and masked and unmasked labels) and self._df_train_labeled (only rows that stayed labeled)
        """

        df_masked, df_labeled = train_test_split(self._df_train_full, test_size=self.nb_samples_labeled, shuffle=True, stratify=self._df_train_full['Label'])
        self._df_train_labeled = df_labeled.reset_index(drop=True)
        self._df_train_masked = self._df_train_full.copy()
        self._df_train_masked.loc[df_masked.index, 'Label'] = DATA_NO_LABEL

    def get_dataloaders_training(self, cuda_state):

        # TO OVERLOAD

        if cuda_state:
            dataloader_train = DataLoader(self._dataset_train, **DATALOADER_PARAMS_CUDA)
            dataloader_valuation = DataLoader(self._dataset_valuation, **DATALOADER_PARAMS_CUDA)
        else:
            dataloader_train = DataLoader(self._dataset_train, **DATALOADER_PARAMS_NO_CUDA)
            dataloader_valuation = DataLoader(self._dataset_valuation, **DATALOADER_PARAMS_NO_CUDA)

        return dataloader_train, dataloader_valuation

    def get_dataloaders_testing(self, cuda_state):

        # TO OVERLOAD

        if cuda_state:
            dataloader_test = DataLoader(self._dataset_test, **DATALOADER_PARAMS_CUDA)
        else:
            dataloader_test = DataLoader(self._dataset_test, **DATALOADER_PARAMS_NO_CUDA)

        return dataloader_test

    def get_info(self):

        info = f'Data: {self.data}\n'
        info += f'Number of samples: {self.nb_samples}\n'
        info += f'Number of classes: {self.nb_classes}\n'
        info += f'Number of training samples: {self.nb_samples_train}\n'
        info += f'Number of labeled samples: {self.nb_samples_labeled}\n'
        info += 'Percent of labeled samples: {:.2f}%\n'.format(self.percent_labeled * 100)
        info += f'Number of testing samples: {self.nb_samples_test}\n'
        info += 'Percent of testing samples: {:.2f}%'.format(self.percent_test * 100)

        return info
