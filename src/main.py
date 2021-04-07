################################################################################
#   Libraries                                                                  #
################################################################################

import time
import os
import argparse
from distutils.dir_util import copy_tree
from tqdm import tqdm
from vars import *

import methods
import datasets
import models
import optimizer
import utils
import networks


import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim import Adam

import warnings
warnings.filterwarnings('ignore')

################################################################################
#   Functionalities                                                            #
################################################################################

METHODS_IMPLEMENTED = {
    'TemporalEnsembling': methods.TemporalEnsemblingClass,
    'Yolo': None
}

DATASETS_IMPLEMENTED = {
    'MNIST': datasets.DatasetMNIST,
    'CIFAR10': datasets.DatasetCIFAR10,
    'CGvsNI': None
}

MODELS = {
    'MNIST': models.MNISTModel(),
    'CIFAR10': models.VGG('VGG11'),
    'CGvsNI': None
}

################################################################################
#   Argparse                                                                   #
################################################################################

parser = argparse.ArgumentParser(description='Semi-supervised MNIST training and testing')

# Functionalities
parser.add_argument('--train', dest='train', action='store_true')
parser.set_defaults(train=False)
parser.add_argument('--test', dest='test', action='store_true')
parser.set_defaults(test=False)
parser.add_argument('--train-test', dest='train_test', action='store_true')
parser.set_defaults(train_test=False)
parser.add_argument('--params-optim', dest='params_optim', action='store_true')
parser.set_defaults(params_optim=False)
parser.add_argument('--supervised-vs-full', dest='supervised_vs_full', action='store_true')
parser.set_defaults(supervised_vs_full=False)

# Data to use
parser.add_argument('--data', type=str, help='data to use')
parser.add_argument('--dataset_name', type=str, help='name of the saved dataset to use')
parser.add_argument('--img_mode', type=str, help='loading method (RGB or L)')
parser.add_argument('--method', type=str, help='training method')
parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer to use')
parser.add_argument('--train_id', type=int, help='index of the trained model to load for tests. In case of training, gets overwritten')

# Training parameters
parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train (default: 300)')
parser.add_argument('--batch_size', type=int, default=100, help='input batch size for training (default: 100)')
parser.add_argument('--shuffle', type=bool, default=True, help='shuffle bool for train dataset (default: True)')

# Testing paramaters
parser.add_argument('--test_batch_size', type=int, default=50, help='input batch size for testing (default: 50)')
parser.add_argument('--decisive_metric', type=str, default='accuracy', help='deciding metric to use in order to choose optimal model')

# Hardware parameter
parser.add_argument('--log_interval', type=int, default=10, help='how many batches to wait before logging training status')
parser.add_argument('--no-verbose', dest='verbose', action='store_false')
parser.set_defaults(verbose=True)
parser.add_argument('--no-cuda', dest='no_cuda', action='store_true')
parser.set_defaults(no_cuda=False)

args = parser.parse_args()

if args.img_mode == None:
    raise RuntimeError('Please specify img_mode param')

args.TRAIN_STEP = TRAIN_STEP

################################################################################
#   Cuda                                                                       #
################################################################################

args.cuda = not args.no_cuda and torch.cuda.is_available()
if args.cuda:
    cudnn.benchmark = True

if args.cuda:
    kwargs_train = {'batch_size': args.batch_size, 'shuffle': args.shuffle, 'num_workers': 8, 'pin_memory': True}
    kwargs_test = {'batch_size': args.test_batch_size, 'shuffle': args.shuffle, 'num_workers': 8, 'pin_memory': True}
else:
    kwargs_train = {'batch_size': args.batch_size, 'shuffle': args.shuffle}
    kwargs_test = {'batch_size': args.test_batch_size, 'shuffle': args.shuffle}

################################################################################
#   Main                                                                       #
################################################################################


class DataContainer:

    def get_info(self):

        header = f'Data: {self.data}\n'
        header += f'Dataset name: {self.dataset_name}\n'
        header += f'Image mode: {self.img_mode}\n'
        header += f'Number of training images: {self.default_info["nb_imgs_train"]}\n'
        header += f'Number of classes: {self.nb_classes}\n'
        header += 'Percent of labeled samples: {:.2f}\n'.format(self.percent_labeled * 100)
        header += '-----------------------------------------------\n'

        return header

    def __init__(self, data, dataset_name, img_mode, trained_model_path, verbose):

        self.data = data
        self.dataset_name = dataset_name
        self.img_mode = img_mode
        self.trained_model_path = trained_model_path
        self.verbose = verbose

        # Creating Dataloaders
        if data in DATASETS_IMPLEMENTED.keys():
            train_dataset_transforms = TRAIN_TRANSFORMS[self.img_mode]
            train_dataset = DATASETS_IMPLEMENTED[data](self.data,
                                                       self.dataset_name,
                                                       self.img_mode,
                                                       'Train',
                                                       transform=train_dataset_transforms)
            self.train_dataloader = DataLoader(train_dataset, **kwargs_train)
            only_sup_dataset = DATASETS_IMPLEMENTED[self.data](self.data,
                                                               self.dataset_name,
                                                               self.img_mode,
                                                               'Only Supervised',
                                                               transform=train_dataset_transforms)
            self.only_sup_dataloader = DataLoader(only_sup_dataset, **kwargs_train)
            valuation_dataset = DATASETS_IMPLEMENTED[self.data](self.data,
                                                                self.dataset_name,
                                                                self.img_mode,
                                                                'Valuation',
                                                                transform=train_dataset_transforms)
            self.valuation_dataloader = DataLoader(valuation_dataset, **kwargs_train)

            test_dataset_transforms = TEST_TRANSFORMS[self.img_mode]
            test_dataset = DATASETS_IMPLEMENTED[self.data](self.data,
                                                           self.dataset_name,
                                                           self.img_mode,
                                                           'Test',
                                                           transform=test_dataset_transforms)
            self.test_dataloader = DataLoader(test_dataset, **kwargs_test)
        else:
            raise RuntimeError(f'Dataset object not implemented for {self.data}')

        # Useful variables
        self.default_info = {
            'nb_imgs_train': len(train_dataset),
            'nb_batches': len(self.train_dataloader)
        }

        self.only_sup_info = {
            'nb_imgs_train': len(only_sup_dataset),
            'nb_batches': len(self.only_sup_dataloader)
        }

        self.nb_classes = train_dataset.nb_classes
        self.percent_labeled = train_dataset.percent_labeled
        self.nb_img_test = len(test_dataset)

        # Saving all relevant info
        header = self.get_info()
        if not os.path.exists(os.path.join(trained_model_path, 'info.txt')):
            with open(os.path.join(trained_model_path, 'info.txt'), 'w+') as f:
                f.write(header)
        if verbose:
            print(header)


class ModelContainer:

    def get_info_model_optim(self):

        header = f'Train_id: {self.train_id}\n'
        header += str(self.model)
        header += f'\nInit mode: {self.init_mode}\n'
        header += utils.get_nb_parameters(self.model)
        header += '-----------------------------------------------\n'
        header += self.optimizer_wrapper.get_info()
        header += '-----------------------------------------------\n'

        return header

    def __init__(self, data_container, train_id, optimizer_name, trained_model_path, pretrained, verbose, cuda):

        # Unpacking data container
        self.data = data_container.data
        self.dataset_name = data_container.dataset_name
        self.img_mode = data_container.img_mode

        self.train_dataloader = data_container.train_dataloader
        self.only_sup_dataloader = data_container.only_sup_dataloader
        self.valuation_dataloader = data_container.valuation_dataloader
        self.test_dataloader = data_container.test_dataloader

        self.default_info = data_container.default_info
        self.only_sup_info = data_container.only_sup_info

        self.nb_classes = data_container.nb_classes
        self.percent_labeled = data_container.percent_labeled
        self.nb_img_test = data_container.nb_img_test

        self.train_id = train_id
        self.optimizer_name = optimizer_name
        self.trained_model_path = trained_model_path
        self.verbose = verbose
        self.cuda = cuda

        # Creating model
        if self.data in MODELS.keys():
            self.model = MODELS[self.data]
            if pretrained:
                self.logs_path = os.path.join(self.trained_model_path, 'logs')
                latest_log, self.start_epoch_id = utils.get_latest_log(self.logs_path)
                checkpoint = torch.load(os.path.join(self.logs_path, latest_log))
                self.model.load_state_dict(checkpoint['state_dict'])
                self.init_mode = 'Pretrained'
            else:
                self.start_epoch_id = 0
                self.init_mode = networks.init_weights(self.model, verbose, init_type='normal')
            if self.cuda:
                self.model = self.model.cuda()
        else:
            raise RuntimeError(f'Model object not implemented for {self.data}')

        # Optimizer with scheduled learning rate
        self.optimizer_wrapper = optimizer.OptimizerWrapper(self.optimizer_name)

    def train(self, mode, method, hyperparameters, epochs):

        if mode == 'default':
            train_dataloader = self.train_dataloader
            nb_img_train = self.default_info['nb_imgs_train']
            nb_batches = self.default_info['nb_batches']
        elif mode == 'only_supervised':
            train_dataloader = self.only_sup_dataloader
            nb_img_train = self.only_sup_info['nb_imgs_train']
            nb_batches = self.only_sup_info['nb_batches']

        # Creating training method
        if method in METHODS_IMPLEMENTED.keys():
            method_container = METHODS_IMPLEMENTED[method](hyperparameters, nb_img_train, self.nb_classes, self.percent_labeled,
                                                           nb_batches, args.batch_size, self.verbose, self.cuda)
        else:
            raise RuntimeError(f'Method not implemented: {method}')

        # Saving all relevant info
        infos_model_optim = self.get_info_model_optim()
        infos_method = method_container.get_info()
        save_str = infos_model_optim + infos_method + f'-----------------------------------------------\nEpochs: {epochs}\n'
        with open(os.path.join(self.trained_model_path, 'info.txt'), 'a+') as f:
            f.write(save_str)
        if self.verbose:
            print(save_str + f'Cuda: {self.cuda}\n')

        method_container.train(train_dataloader, self.valuation_dataloader, self.model, self.optimizer_wrapper,
                               nb_img_train, self.nb_classes, nb_batches, args.batch_size, epochs,
                               self.trained_model_path, self.start_epoch_id, self.verbose)  # Doesn't return anything, just saves all relevant data its dedicated folder

    def test(self):

        # Creating testing class
        test_method = methods.TestingClass(self.verbose, self.cuda)
        report, self.metrics = test_method.test(self.test_dataloader, self.model, TEST_RUNS)

        if self.verbose:
            print(report)
        report = f'Number of test runs: {TEST_RUNS}\n' + report
        with open(os.path.join(self.trained_model_path, f'results.txt'), 'w+') as f:
            f.write(report)


def main():

    if not args.test:
        if args.train_id == None:
            args.pretrained = False
            args.train_id = utils.get_train_id(TRAINED_MODELS_PATH)
        else:
            args.pretrained = True
        args.full_name = str(args.train_id) + '_' + args.dataset_name + '_' + args.method

    else:
        args.pretrained = True
        if args.train_id != None:
            args.full_name = utils.get_trained_model_from_id(args.train_id)
        else:
            raise RuntimeError('Please provide a train_id in order to run tests')

    args.trained_model_path = os.path.join(TRAINED_MODELS_PATH, args.full_name)
    if not os.path.exists(args.trained_model_path):
        os.makedirs(args.trained_model_path)

    data_container = DataContainer(args.data, args.dataset_name, args.img_mode, args.trained_model_path, args.verbose)

    if args.train:  # Trains one model

        print('Starting training...')

        model_container = ModelContainer(data_container, args.train_id, args.optimizer, args.trained_model_path, args.pretrained, args.verbose, args.cuda)

        train_mode = 'default'
        model_container.train(train_mode, args.method, HYPERPARAMETERS_DEFAULT[args.method], args.epochs)

        print('Training done!')

    # -----------------------------

    if args.test:  # Tests one trained model

        print('Running tests...')

        model_container = ModelContainer(data_container, args.train_id, args.optimizer, args.trained_model_path, args.pretrained, args.verbose, args.cuda)

        model_container.test()

        print('Tests done!')

    # -----------------------------

    if args.train_test:  # Trains then tests one model

        print('Starting training...')

        model_container = ModelContainer(data_container, args.train_id, args.optimizer, args.trained_model_path, args.pretrained, args.verbose, args.cuda)

        train_mode = 'default'
        model_container.train(train_mode, args.method, HYPERPARAMETERS_DEFAULT[args.method], args.epochs)

        print('\nRunning tests...')

        model_container.test()

        print('Tests done!')

    # -----------------------------

    if args.params_optim:  # From a number of hyperparamters, returns the set that goves the best performance

        print('Hyperparamters optimization...')

        args.pretrained = False
        list_hyperparameters = utils.get_hyperparameters_combinations(args.method)
        print(f'Testing {len(list_hyperparameters)} combinations')

        best_model_id = 0
        best_model_metrics = {}
        for metric_funct in METRICS.keys():
            best_model_metrics[metric_funct] = 0.

        train_mode = 'default'
        for i in range(len(list_hyperparameters)):

            sub_model_path = os.path.join(args.trained_model_path, str(i + 1))
            if not os.path.exists(sub_model_path):
                os.makedirs(sub_model_path)

            model_container = ModelContainer(data_container, args.train_id, args.optimizer, sub_model_path, args.pretrained, args.verbose, args.cuda)

            model_container.train(train_mode, args.method, list_hyperparameters[i], args.epochs)
            model_container.test()

            if model_container.metrics[args.decisive_metric] > best_model_metrics[args.decisive_metric]:
                best_model_id = i
                best_model_metrics = model_container.metrics

        copy_tree(os.path.join(args.trained_model_path, best_model_id), os.path.join(args.trained_model_path, 'best-model'))

        print(f'Best model id: {best_model_id}')
        print('Search done!')

    if args.supervised_vs_full:

        args.pretrained = False

        # -----------------------------

        print('Training only on the supervised part of the dataset...')

        subfolder = 'only_supervised'
        train_mode = 'only_supervised'
        list_metrics_only_sup = []

        if args.verbose:
            pbar = tqdm(range(ONLY_SUP_RUNS))
        else:
            pbar = range(ONLY_SUP_RUNS)

        for i in pbar:

            sub_model_path = os.path.join(args.trained_model_path, subfolder, str(i + 1))
            if not os.path.exists(sub_model_path):
                os.makedirs(sub_model_path)

            model_container = ModelContainer(data_container, args.train_id, args.optimizer, sub_model_path, args.pretrained, False, args.cuda)

            model_container.train(train_mode, args.method, HYPERPARAMETERS_DEFAULT[args.method], args.epochs)
            model_container.test()

            list_metrics_only_sup.append(model_container.metrics)

        avg_metrics_only_sup = utils.get_avg_metrics(list_metrics_only_sup)
        avg_report_only_sup = utils.get_metrics_report(avg_metrics_only_sup)

        with open(os.path.join(args.trained_model_path, subfolder, 'metrics.txt'), 'w+') as f:
            f.write(avg_report_only_sup)

        if args.verbose:
            print(avg_report_only_sup)

        # -----------------------------

        print('Training only on the supervised part of the dataset, without the unsupervised loss...')

        hyperparameters_no_unsup_loss = HYPERPARAMETERS_DEFAULT[args.method]
        hyperparameters_no_unsup_loss['unsup_loss_max_weight'] = 0.

        subfolder = 'only_supervised_no_unsup_loss'
        train_mode = 'only_supervised'
        list_metrics_only_sup_no_unsup_loss = []

        if args.verbose:
            pbar = tqdm(range(ONLY_SUP_RUNS))
        else:
            pbar = range(ONLY_SUP_RUNS)

        for i in pbar:

            sub_model_path = os.path.join(args.trained_model_path, subfolder, str(i + 1))
            if not os.path.exists(sub_model_path):
                os.makedirs(sub_model_path)

            model_container = ModelContainer(data_container, args.train_id, args.optimizer, sub_model_path, args.pretrained, False, args.cuda)

            model_container.train(train_mode, args.method, hyperparameters_no_unsup_loss, args.epochs)
            model_container.test()

            list_metrics_only_sup_no_unsup_loss.append(model_container.metrics)

        avg_metrics_only_sup_no_unsup_loss = utils.get_avg_metrics(list_metrics_only_sup_no_unsup_loss)
        avg_report_only_sup_no_unsup_loss = utils.get_metrics_report(avg_metrics_only_sup_no_unsup_loss)

        with open(os.path.join(args.trained_model_path, subfolder, 'metrics.txt'), 'w+') as f:
            f.write(avg_report_only_sup_no_unsup_loss)

        if args.verbose:
            print(avg_report_only_sup_no_unsup_loss)

        # -----------------------------

        print('Training on all of the dataset')

        subfolder = 'full_dataset'
        train_mode = 'default'
        sub_model_path = os.path.join(args.trained_model_path, subfolder)
        if not os.path.exists(sub_model_path):
            os.makedirs(sub_model_path)
            
        model_container = ModelContainer(data_container, args.train_id, args.optimizer, sub_model_path, args.pretrained, False, args.cuda)

        model_container.train(train_mode, args.method, HYPERPARAMETERS_DEFAULT[args.method], args.epochs)
        model_container.test()

        if args.verbose:
            print(utils.get_metrics_report(model_container.metrics))


if __name__ == '__main__':
    main()
