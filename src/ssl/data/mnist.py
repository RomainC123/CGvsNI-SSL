################################################################################
#   Libraries                                                                  #
################################################################################

import os
import pandas as pd
import torchvision.transforms as transforms

from .image import ImageDatasetContainer
from ..utils.paths import DATASETS_PATH
from ..utils.transforms import IMAGE_TRANSFORMS_TRAIN, IMAGE_TRANSFORMS_TEST

################################################################################
#   Image container class                                                      #
################################################################################


class MNISTDatasetContainer(ImageDatasetContainer):

    def __init__(self, data, nb_samples_total, nb_samples_test, nb_samples_labeled, cuda_state, **kwargs):

        super(MNISTDatasetContainer, self).__init__(data, nb_samples_total, nb_samples_test, nb_samples_labeled, **kwargs)

    def _get_data(self):

        return pd.read_csv(os.path.join(DATASETS_PATH, self.data, 'dataset.csv'))

    def _get_transforms(self):

        return IMAGE_TRANSFORMS_TRAIN[self.data][self.img_mode], IMAGE_TRANSFORMS_TEST[self.data][self.img_mode]
