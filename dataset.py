import os
import pathlib
import pandas as pd
import random

from sklearn.model_selection import train_test_split


ROOT_PATH = os.path.join(pathlib.Path(__file__).parent.absolute(), 'datasets')
DATA_PATH = os.path.join(ROOT_PATH, 'raw')
OUTPUT_PATH = os.path.join(ROOT_PATH, 'clean')

DATASET_DICT = {"Artlantis": 1,  # 1 => CGI dataset
                "Autodesk": 1,
                "Corona": 1,
                "VRay": 1,
                "RAISE": 2,  # 2 => NI dataset
                "VISION": 2}


def get_nb_imgs_set(nb_cgi_sets, nb_ni_sets, size, balance):

    if nb_cgi_sets == 0 and nb_ni_sets == 0:
        raise ValueError('Please choose at least one set')
    else:
        if nb_cgi_sets == 0:
            nb_imgs_CGI_set = 0.
        else:
            nb_imgs_CGI_set = size * balance / (nb_cgi_sets * 4)

        if nb_ni_sets == 0:
            nb_imgs_NI_set = 0.
        else:
            nb_imgs_NI_set = size * (1 - balance) / nb_ni_sets

    return nb_imgs_CGI_set, nb_imgs_NI_set


def good_parameters(nb_imgs_CGI_set, nb_imgs_NI_set, test_size, nb_labels):

    return (nb_imgs_CGI_set.is_integer() and
            nb_imgs_NI_set.is_integer() and
            (nb_imgs_CGI_set * (1 - test_size) * nb_labels).is_integer() and
            (nb_imgs_NI_set * (1 - test_size) * nb_labels).is_integer())


def multiply_list_elems(list, nb_multiples):

    multiple_list = []

    for x in list:
        for i in range(nb_multiples):
            multiple_list.append(x)

    return multiple_list


def make_frame_set(set, img_type, data_type, nb_imgs, nb_multiples, test_size, nb_labels, ):

    if img_type == 'CG':
        label = 1
        multiple = nb_multiples[0]
    elif img_type == 'N':
        label = 2
        multiple = nb_multiples[1]
    else:
        raise ValueError(f'Unknown img_type: {img_type}')

    img_folder_path = os.path.join(DATA_PATH, set, data_type)

    if not os.path.isdir(img_folder_path):
        raise ValueError(f'Unknown dataset name: {img_folder_path}')
    else:
        list_imgs = os.listdir(img_folder_path)[:int(nb_imgs)]  # No need to shuffle this list as os.listdir is already random

    train_imgs, test_imgs = train_test_split(list_imgs, test_size=test_size, shuffle=False)

    df_imgs = pd.DataFrame(columns=['Name', 'Label', 'Test'])
    df_imgs['Name'] = multiply_list_elems(train_imgs + test_imgs, multiple)
    df_imgs['Label'] = multiply_list_elems(random.sample([label for x in range(int(len(train_imgs) * nb_labels))] +
                                                         [0 for x in range(int(len(train_imgs) * nb_labels), len(train_imgs))], len(train_imgs)) + ['Nan' for x in range(len(test_imgs))], multiple)
    df_imgs['Test'] = multiply_list_elems([False for x in range(len(train_imgs))] + [True for x in range(len(test_imgs))], multiple)

    return df_imgs


def make_dataset(CGI_sets, NI_sets, size=4000, nb_multiples=(4, 1), balance=0.5, test_size=0.1, nb_labels=0.1, shuffle=True, data_type='data_512crop'):
    """
    Grabs images names and creates a list of training samples and testing
    samples, and saves it in a .csv file
    All chosen CGIs are repeated 4 times (to account for the fact that there are
    4 times as much NIs than CGIs)
    Images are chosen in a balanced manner from all available sets corresponding
    to each label

    Args:
        - CGI_sets (list): list containing the list of name of the CGIs to use
        for the dataset creation
        - NI_sets (list): list containing the list of name of the NIs to use for
        the dataset creation
        - size (int, default 1000): final number of images for the dataset
        - nb_multiples (tuple, default (4, 1)): (nb of multiples of each CGIs,
        nb of multiples of each NI)
        - balance (float, default 0.5): percentage of CGIs in the final dataset
        - test_size (float, default 0.1): share of each set to be used for
        testing
        - nb_labels (float, default 1.): share of the training samples to save
        with label (for semi-supervised learning), set to 1. for supervised
        training
        - shuffle (bool, default True): whether or not to shuffle the images
        before making the split
        - data_type (str, default 'data_512crop'): target folder for each
        dataset
    """

    nb_cgi_sets = len(CGI_sets)
    nb_ni_sets = len(NI_sets)

    nb_imgs_CGI_set, nb_imgs_NI_set = get_nb_imgs_set(nb_cgi_sets, nb_ni_sets, size, balance)

    if not good_parameters(nb_imgs_CGI_set, nb_imgs_NI_set, test_size, nb_labels):
        # Guarantees that all numbers end up as integers
        raise ValueError('Non-integer sizes found, please check your values or use default ones')

    dataset_name = f"{'_'.join(CGI_sets + NI_sets)}_{str(size)}_{str(balance)}_{str(test_size)}_{str(nb_labels)}_{data_type}.csv"
    dataset_path = os.path.join(OUTPUT_PATH, dataset_name)

    if os.path.exists(dataset_path):
        raise NameError('Dataset already exists')
    else:
        with open(dataset_path, 'w') as f:
            f.write('Name,Label,Test\n')

    for cgi_set in CGI_sets:
        df_imgs = make_frame_set(cgi_set, 'CG', data_type, nb_imgs_CGI_set, nb_multiples, test_size, nb_labels)
        with open(dataset_path, 'a') as f:
            f.write(df_imgs.to_csv(header=False))

    for ni_set in NI_sets:
        df_imgs = make_frame_set(ni_set, 'N', data_type, nb_imgs_NI_set, nb_multiples, test_size, nb_labels)
        with open(dataset_path, 'a') as f:
            f.write(df_imgs.to_csv(header=False))


CGI_sets = ['Artlantis']
NI_sets = ['Autodesk']
make_dataset(CGI_sets, NI_sets)
