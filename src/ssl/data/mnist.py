################################################################################
#   Libraries                                                                  #
################################################################################

import os
import pandas as pd
import torch
import torchvision.transforms as transforms

from PIL import Image
from torch.utils.data.sampler import RandomSampler, BatchSampler

from .image import ImageDatasetContainer
from ..utils.paths import DATASETS_PATH
from ..utils.transforms import IMAGE_TRANSFORMS_TRAIN, IMAGE_TRANSFORMS_TEST

################################################################################
#   Image container class                                                      #
################################################################################


class MNISTDatasetContainer(ImageDatasetContainer):

    def __init__(self, data, nb_samples_total, nb_samples_test, nb_samples_labeled, cuda_state, **kwargs):

        self.cuda_state = cuda_state

        super(MNISTDatasetContainer, self).__init__(data, nb_samples_total, nb_samples_test, nb_samples_labeled, **kwargs)

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

        sampler_train = RandomSampler(self._dataset_train)
        self._batch_sampler_train = BatchSampler(sampler_train, self.batch_size, drop_last=True)
