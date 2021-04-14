################################################################################
#   Libraries                                                                  #
################################################################################

import os
import pathlib
import math
import random
import re
import argparse
import pandas as pd

from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')

ROOT_PATH = pathlib.Path(__file__).resolve().parents[1].absolute()

DATASETS_PATH = os.path.join(ROOT_PATH, 'datasets')
if not os.path.exists(DATASETS_PATH):
    raise RuntimeError('Please create datasets folder and add data to it')

################################################################################
#   Argparse                                                                   #
################################################################################

parser = argparse.ArgumentParser(description='Dataset maker')

parser.add_argument('--data', type=str, help='type of data to use for creating the dataset')
parser.add_argument('--dataset_name', type=str, help='name of the created dataset')

parser.add_argument('--dataset_size', type=int, default=-1, help='desired number of images in dataset (train and test) (default: -1)')
parser.add_argument('--val_size', type=int, default=1000, help='desired number of images in valuation set, only for display purposes (default: 1000)')
parser.add_argument('--test_size', type=float, default=0.2, help='percent of samples to be used for testing (default: 0.2)')
parser.add_argument('--use-presplit', dest='use_presplit', action='store_true')
parser.set_defaults(use_presplit=False)

parser.add_argument('--nb_labels', type=float, default=1000, help='number of labelized train samples (defautl: 1000)')

args = parser.parse_args()

################################################################################
#   Main dataset maker class                                                   #
################################################################################


class DatasetMaker:

    def good_parameters(self):

        assert (self.dataset_size * self.test_size).is_integer()
        assert (self.dataset_size * (1 - self.test_size) * self.nb_labels).is_integer()

    def get_even_class_labels(self, list_labels, nb_labels):

        id_kept = list(range(len(list_labels)))
        labels_ids = {}

        for i in range(len(list_labels)):
            if list_labels[i] not in labels_ids.keys():
                labels_ids[list_labels[i]] = []
            labels_ids[list_labels[i]].append(i)

        unique_labels = labels_ids.keys()

        sum = 0
        cnt = 0
        labels_per_class = nb_labels / len(unique_labels)
        upper_labels = math.ceil(labels_per_class)
        lower_labels = math.floor(labels_per_class)
        even_labels_list = []
        for label in unique_labels:
            if sum + upper_labels + (len(unique_labels) - cnt - 1) * lower_labels <= nb_labels:
                even_labels_list += random.sample(labels_ids[label], upper_labels)
                sum += upper_labels
            else:
                even_labels_list += random.sample(labels_ids[label], lower_labels)
                sum += lower_labels
            cnt += 1

        return even_labels_list

    def get_masked_labels(self, list_labels, list_unmasked_labels):

        for i in range(len(list_labels)):
            if i not in list_unmasked_labels:
                list_labels[i] = -1

        return list_labels

    def get_valuation_bool(self, list_labels, list_valuation_idx):

        return [i in list_valuation_idx for i in range(len(list_labels))]

    def get_images(self):
        # To overload
        pass

    def get_train_test(self):
        # To overload
        pass

    def __init__(self, args):

        self.dataset_name = args.dataset_name
        self.dataset_size = args.dataset_size
        self.val_size = args.val_size
        self.test_size = args.test_size
        self.use_presplit = args.use_presplit
        self.nb_labels = args.nb_labels

        if self.use_presplit:
            self.dataset_size = -1
            self.test_size = 0.

        self.raw_path = os.path.join(DATASETS_PATH, self.data, 'raw')
        if not os.path.exists(self.raw_path):
            raise RuntimeError('Please create raw folder and populate it')

        self.clean_path = os.path.join(DATASETS_PATH, self.data, 'clean')
        if not os.path.exists(self.clean_path):
            os.makedirs(sefl.clean_path)

        if self.dataset_name == None:
            # default naming convention
            if self.use_presplit:
                self.dataset_name = f"{self.data}_{str(self.nb_labels)}.csv"
            else:
                self.dataset_name = f"{self.data}_{str(self.dataset_size)}_{str(self.test_size)}_{str(self.nb_labels)}.csv"
        else:
            self.dataset_name = self.dataset_name + '.csv'
        self.dataset_path = os.path.join(self.clean_path, self.dataset_name)

    def make(self):

        print('Creating dataset...')

        self.get_images()
        self.good_parameters()
        self.get_train_test()

        self.train_imgs.reset_index(drop=True, inplace=True)
        self.test_imgs.reset_index(drop=True, inplace=True)

        # Column containing unmasked labels for all data
        self.train_imgs['real_label'] = self.train_imgs['Label']
        self.test_imgs['real_label'] = self.test_imgs['Label']

        list_valuation_idx = self.get_even_class_labels(self.train_imgs['Label'], self.val_size)
        self.train_imgs['valuation'] = self.get_valuation_bool(self.train_imgs['Label'], list_valuation_idx)
        self.test_imgs['valuation'] = [False for i in range(len(self.test_imgs['Label']))]

        # Column containing masked data for the train set, and unmasked data for the tests set, as this column isn't used for the testing set
        list_unmasked_labels = self.get_even_class_labels(self.train_imgs['Label'], self.nb_labels)
        self.train_imgs['train_label'] = self.get_masked_labels(self.train_imgs['Label'], list_unmasked_labels)
        self.test_imgs['train_label'] = self.test_imgs['Label']

        self.df_data = pd.concat([self.train_imgs, self.test_imgs]).reset_index(drop=True)
        self.df_data.drop('Label', axis=1, inplace=True)
        self.df_data.insert(4, "Test", [False for x in range(len(self.train_imgs))] + [True for x in range(len(self.test_imgs))])

        self.df_data = self.df_data[['Name', 'real_label', 'train_label', 'valuation', 'Test']]

        if os.path.exists(self.dataset_path):
            raise NameError('Dataset already exists')
        else:
            with open(self.dataset_path, 'w+') as f:
                f.write('Name,Real label,Train label,Val,Test\n')
                f.write(self.df_data.to_csv(header=False))

        print('Dataset created: ', self.dataset_name)
        print('Done!')

################################################################################
#   Children classes                                                           #
################################################################################


class MNISTDatasetMaker(DatasetMaker):

    def get_images(self):

        if self.dataset_size == -1:
            self.df_imgs = pd.read_csv(os.path.join(self.raw_path, 'name_labels.csv'))
            self.dataset_size = len(self.df_imgs)
        else:
            self.df_imgs = pd.read_csv(os.path.join(self.raw_path, 'name_labels.csv')).iloc[:self.dataset_size]

    def get_train_test(self):

        self.train_imgs, self.test_imgs = train_test_split(self.df_imgs, test_size=self.test_size, shuffle=True, stratify=True)

    def __init__(self, args):

        self.data = 'MNIST'
        super().__init__(args)


class CIFAR10DatasetMaker(DatasetMaker):

    def get_images(self):

        if self.dataset_size == -1:
            self.df_imgs = pd.read_csv(os.path.join(self.raw_path, 'name_labels.csv'))
            self.dataset_size = len(self.df_imgs)
        else:
            self.df_imgs = pd.read_csv(os.path.join(self.raw_path, 'name_labels.csv')).iloc[:self.dataset_size]
        self.df_imgs['Label'] = self.df_imgs['Label'].apply(lambda x: int(x[1]))

    def get_train_test(self):

        if not self.use_presplit:
            self.train_imgs, self.test_imgs = train_test_split(self.df_imgs[['Name', 'Label']], test_size=self.test_size, shuffle=True)
        else:
            self.train_imgs = self.df_imgs.loc[~self.df_imgs['Test']][['Name', 'Label']]
            self.test_imgs = self.df_imgs.loc[self.df_imgs['Test']][['Name', 'Label']]

    def __init__(self, args):

        self.data = 'CIFAR10'
        super().__init__(args)

################################################################################
#   Children classes                                                           #
################################################################################


DATASET_CLASSES = {
    'MNIST': MNISTDatasetMaker,
    'CIFAR10': CIFAR10DatasetMaker
}


def make_dataset(args):

    if args.data in DATASET_CLASSES.keys():
        builder = DATASET_CLASSES[args.data](args)
        builder.make()


if __name__ == '__main__':
    make_dataset(args)
