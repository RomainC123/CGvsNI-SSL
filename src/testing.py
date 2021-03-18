################################################################################
#   Libraries                                                                  #
################################################################################

import os
import pathlib
import pickle
import argparse
import numpy as np

import temporal_ensembling
import datasets
import models
# import display
import utils
from vars import *

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

from tqdm import tqdm

################################################################################
#   Argparse                                                                   #
################################################################################

parser = argparse.ArgumentParser(description='Semi-supervised MNIST testing')

# Data to use
parser.add_argument('--data', type=str, help='data to use')
parser.add_argument('--dataset_name', type=str, help='name of the saved dataset to use')
parser.add_argument('--img_mode', type=str, help='loading method (RGB or L)')
parser.add_argument('--method', type=str, help='type of training used')
parser.add_argument('--train_id', type=int, help='index of the trained model to load for tests. In case of training, gets overwritten')

# Testing paramaters
parser.add_argument('--test_runs', type=int, default=2, help='number of test runs used to compute avg')
parser.add_argument('--test_batch_size', type=int, default=50, help='input batch size for testing (default: 50)')

# Whether or not to show examples of images with true labels and prediction
parser.add_argument('--example', dest='example', action='store_true')
parser.add_argument('--no-example', dest='example', action='store_false')
parser.set_defaults(example=False)
# Look for best params from a set of trainings
parser.add_argument('--best-params', dest='best_params', action='store_true')
parser.add_argument('--no-best-params', dest='best_params', action='store_false')
parser.set_defaults(best_params=False)

# Hardware parameter
parser.add_argument('--log_interval', type=int, default=10, help='how many batches to wait before logging training status')
parser.add_argument('--no_cuda', default=False, help='disables CUDA training')

args = parser.parse_args()

if args.img_mode == None:
    raise RuntimeError('Please specify img_mode param')

################################################################################
#   Cuda                                                                       #
################################################################################

args.cuda = not args.no_cuda and torch.cuda.is_available()
if args.cuda:
    cudnn.benchmark = True

if args.cuda:
    kwargs = {'batch_size': args.test_batch_size, 'shuffle': False, 'num_workers': 8, 'pin_memory': True}
else:
    kwargs = {'batch_size': args.test_batch_size, 'shuffle': False}

################################################################################
#   Testing                                                                    #
################################################################################


def main():

    def get_dataset_caracs(dataset, dataset_name):
        """
        Given a dataset object, returns the number of train samples, test samples, and the percentage of labeled samples in the test set
        """

        dataset_train = dataset(args, False)
        dataset_test = dataset(args, True)
        nb_train, percent_labeled = len(dataset_train), dataset_train.percent_labeled
        nb_test = len(dataset_test)

        return nb_train, nb_test, percent_labeled

    def get_test_report(test_dataloader, model, args):

        list_classification_reports = []
        for i in range(args.test_runs):
            pred_labels, real_labels = METHODS_IMPLEMENTED[args.method]['testing'](test_dataloader, model, False, args)
            list_classification_reports.append(classification_report(real_labels, pred_labels, digits=3, output_dict=True))
        full_classification_report = utils.avg_classifications_reports(list_classification_reports)
        return full_classification_report

    print('Running tests...')

    # Creating all the relevant paths
    args.logs_path_full = os.path.join(LOGS_PATH, args.data, args.dataset_name, args.method + '_' + str(args.train_id))
    if not os.path.exists(args.logs_path_full):
        raise RuntimeError('Trained model not found, please check the given id')

    args.results_path = os.path.join(TEST_RESULTS_PATH, args.data, args.dataset_name)
    if not os.path.exists(args.results_path):
        os.makedirs(args.results_path)

    args.nb_train, args.nb_test, args.percent_labeled = get_dataset_caracs(DATASETS_IMPLEMENTED[args.data], args)

    # Creating Dataset object
    if args.data in DATASETS_IMPLEMENTED.keys():
        test_dataset_transforms = TRANSFORMS_TEST[args.data]
        test_dataset = DATASETS_IMPLEMENTED[args.data](args,
                                                       True,
                                                       transform=test_dataset_transforms)
        test_dataloader = DataLoader(test_dataset, **kwargs)

        model = MODELS[args.data]
    else:
        raise RuntimeError(f'Data type not implemented: {args.data}')

    print('Image mode: ', args.img_mode)

    # Useful variables
    args.nb_img_test = len(test_dataset)
    args.nb_batches_test = len(test_dataloader)

    print('Number of test data: {}'.format(len(test_dataloader.dataset)))

    if args.method in METHODS_IMPLEMENTED:
        if args.best_params:
            for param_id in sorted(os.listdir(args.logs_path_full)):
                args.logs_path_temp = os.path.join(args.logs_path_full, str(param_id))
                latest_log = utils.get_latest_log(args.logs_path_temp)
                checkpoint = torch.load(os.path.join(args.logs_path_temp, latest_log))
                model.load_state_dict(checkpoint['state_dict'])
                args.full_classification_report = get_test_report(test_dataloader, model, args)
                utils.save_results(param_id, args)
        else:
            latest_log = utils.get_latest_log(args.logs_path_full)
            checkpoint = torch.load(os.path.join(args.logs_path_full, latest_log))
            model.load_state_dict(checkpoint['state_dict'])

    print('Tests done!')

    # Showing 9 examples of predictions
    if args.example:
        display.classification_display(test_dataset, model, args)


if __name__ == '__main__':
    main()
