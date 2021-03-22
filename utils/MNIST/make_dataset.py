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

################################################################################
#   Argparse                                                                   #
################################################################################

parser = argparse.ArgumentParser(description='Dataset maker')

parser.add_argument('--dataset_name', type=str, help='name of the saved dataset to use')

parser.add_argument('--dataset_size', type=int, default=-1, help='desired number of images in dataset (train and test) (default: 30000)')
parser.add_argument('--test_size', type=float, default=0.2, help='percent of samples to be used for testing (default: 0.2)')
parser.add_argument('--nb_labels', type=float, default=0.1, help='percent of test samples to be labelized (defautl: 0.1)')

args = parser.parse_args()

################################################################################
#   Paths                                                                      #
################################################################################

ROOT_PATH = pathlib.Path(__file__).resolve().parents[2].absolute()

DATA_PATH = os.path.join(ROOT_PATH, 'datasets', 'MNIST')
if not os.path.exists(DATA_PATH):
    raise RuntimeError('Please create datasets folder and add data to it')

CLEAN_PATH = os.path.join(DATA_PATH, 'clean')
if not os.path.exists(CLEAN_PATH):
    os.makedirs(CLEAN_PATH)

RAW_PATH = os.path.join(DATA_PATH, 'raw')
if not os.path.exists(RAW_PATH):
    raise RuntimeError('Please create raw folder and populate it')

################################################################################
#   Dataset maker                                                              #
################################################################################


def good_parameters(nb_imgs, test_size, nb_labels):

    assert (nb_imgs * test_size).is_integer()
    assert (nb_imgs * (1 - test_size) * nb_labels).is_integer()


def mask_labels(list_labels, nb_labels):

    id_kept = list(range(len(list_labels)))
    labels_ids = {}

    for i in range(len(list_labels)):
        if list_labels[i] not in labels_ids.keys():
            labels_ids[list_labels[i]] = []
        labels_ids[list_labels[i]].append(i)

    unique_labels = labels_ids.keys()

    sum = 0
    cnt = 0
    labels_per_class = len(list_labels) * nb_labels / len(unique_labels)
    upper_labels = math.ceil(labels_per_class)
    lower_labels = math.floor(labels_per_class)
    labels_kept = []
    for label in unique_labels:
        if sum + upper_labels + (len(unique_labels) - cnt - 1) * lower_labels <= int(len(list_labels) * nb_labels):
            labels_kept += random.sample(labels_ids[label], upper_labels)
            sum += upper_labels
        else:
            labels_kept += random.sample(labels_ids[label], lower_labels)
            sum += lower_labels
        cnt += 1

    assert len(labels_kept) == len(list_labels) * nb_labels

    cnt = 0
    for i in range(len(list_labels)):
        if i not in labels_kept:
            list_labels[i] = -1
            cnt += 1

    return list_labels


def make_dataset(dataset_size, test_size, nb_labels, name=None):
    """
    Grabs images names and creates a list of training samples and testing
    samples, and saves it in a .csv file
    Args:
        - name (str, default None): name override for the final file
        - size (int, default -1): number of images for the dataset, use -1 to use all avaliable images
        - test_size (float, default 0.2): percent of the train size to be used
        as test size
        - nb_labels (float, default 0.1): share of the training samples to save
        with label (for semi-supervised learning), set to 1. for supervised
        training
    """

    if name == None:
        # default naming convention
        dataset_name = f"mnist_{str(dataset_size)}_{str(test_size)}_{str(nb_labels)}.csv"
    else:
        dataset_name = name + '.csv'
    dataset_path = os.path.join(CLEAN_PATH, dataset_name)

    print('Creating dataset...')

    if dataset_size == -1:
        df_imgs = pd.read_csv(os.path.join(RAW_PATH, 'name_labels.csv'))
        dataset_size = len(df_imgs)
    else:
        df_imgs = pd.read_csv(os.path.join(RAW_PATH, 'name_labels.csv')).iloc[:dataset_size]

    good_parameters(dataset_size, test_size, nb_labels)

    train_imgs, test_imgs = train_test_split(df_imgs, test_size=test_size, shuffle=True)

    train_imgs.reset_index(drop=True, inplace=True)
    test_imgs.reset_index(drop=True, inplace=True)

    # Column containing unmasked labels for all data
    train_imgs['real_label'] = train_imgs['class']
    test_imgs['real_label'] = test_imgs['class']

    # Column containing masked data for the train set, and unmasked data for the tests set, as this column isn't used for the testing set
    train_imgs['train_label'] = mask_labels(train_imgs['class'], nb_labels)
    test_imgs['train_label'] = test_imgs['class']

    df_data = pd.concat([train_imgs, test_imgs]).reset_index(drop=True)
    df_data.drop('class', axis=1, inplace=True)
    df_data.insert(3, "Test", [False for x in range(len(train_imgs))] + [True for x in range(len(test_imgs))])

    if os.path.exists(dataset_path):
        raise NameError('Dataset already exists')
    else:
        with open(dataset_path, 'w+') as f:
            f.write('Name,Real label,Train label,Test\n')
            f.write(df_data.to_csv(header=False))

    print('Dataset created: ', dataset_name)
    print('Done!')


if __name__ == '__main__':
    make_dataset(args.dataset_size, args.test_size, args.nb_labels, args.dataset_name)
