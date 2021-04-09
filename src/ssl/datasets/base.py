import os
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from ..utils.paths import DATASETS_PATH
from ..utils.constants import VAL_NUMBER, DATA_NO_LABEL
from .datasets import ImageDataset
from ..utils.transforms import IMAGE_TRANSFORMS_TRAIN, IMAGE_TRANSFORMS_TEST


class BaseDatasetContainer:

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
        self._df_train_masked['Label'][df_masked.index] = DATA_NO_LABEL

    def get_dataloaders(self, **kwargs):

        # TO OVERLOAD
        # Create datasets first

        self.dataloader_train = DataLoader(dataset_train, **kwargs)
        self.dataloader_valuation = DataLoader(dataset_valuation, **kwargs)
        self.dataloader_test = DataLoader(dataset_test, **kwargs)

    def get_info(self):

        info = f'Data: {self.data}\n'
        info += f'Number of samples: {self.nb_samples}\n'
        info += f'Number of classes: {self.nb_classes}\n'
        info += f'Number of training samples: {self.nb_samples_train}\n'
        info += f'Number of labeled samples: {self.nb_samples_labeled}\n'
        info += 'Percent of labeled samples: {:.2f}%\n'.format(self.percent_labeled * 100)
        info += f'Number of testing samples: {self.nb_samples_test}\n'
        info += 'Percent of testing samples: {:.2f}%\n'.format(self.percent_test * 100)
        info += '-----------------------------------------------\n'

        return info


class ImageDatasetContainer(BaseDatasetContainer):

    def __init__(self, data, nb_samples_test, nb_samples_labeled):
        super(ImageDatasetContainer, self).__init__(data, nb_samples_test, nb_samples_labeled)

    def get_dataloaders(self, img_mode, **kwargs):

        dataset_train = ImageDataset(self.data,
                                     img_mode,
                                     self._df_train_masked,
                                     transform=IMAGE_TRANSFORM_TRAIN[img_mode])
        dataset_valuation = ImageDataset(self.data,
                                         img_mode,
                                         self._df_valuation,
                                         transform=IMAGE_TRANSFORM_TRAIN[img_mode])
        dataset_test = ImageDataset(self.data,
                                    img_mode,
                                    self._df_test,
                                    transform=IMAGE_TRANSFORM_TEST[img_mode])

        super(ImageDatasetContainer, self).get_dataloaders(self, **kwargs)
