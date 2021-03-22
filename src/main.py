################################################################################
#   Libraries                                                                  #
################################################################################

import os
import argparse
from tqdm import tqdm
from vars import *

import methods
import utils
import networks

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim import Adam

METHODS_IMPLEMENTED = {
    'TemporalEnsembling': methods.TemporalEnsemblingClass,
    'Yolo': None
}

################################################################################
#   Argparse                                                                   #
################################################################################

parser = argparse.ArgumentParser(description='Semi-supervised MNIST training')

# Usage
parser.add_argument('--train', dest='train', action='store_true')
parser.set_defaults(train=False)
parser.add_argument('--test', dest='test', action='store_true')
parser.set_defaults(test=False)
parser.add_argument('--params-optim', dest='params_optim', action='store_true')
parser.set_defaults(params_optim=False)

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
parser.add_argument('--test_runs', type=int, default=3, help='number of test runs used to compute avg')
parser.add_argument('--test_batch_size', type=int, default=50, help='input batch size for testing (default: 50)')

# Hardware parameter
parser.add_argument('--log_interval', type=int, default=10, help='how many batches to wait before logging training status')
parser.add_argument('--no_cuda', default=False, help='disables CUDA training')

parser.add_argument('--no-verbose', dest='verbose', action='store_false')
parser.set_defaults(verbose=True)
parser.add_argument('--no-cuda', dest='no_cuda', action='store_true')
parser.set_defaults(no_cuda=False)

args = parser.parse_args()

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


def main():

    def train(args):
        """
        Trains one model
        --------------------------------------
        Inputs:
        - data type and dataset_name
        - method and hyperparamters
        - optimizer
        Outputs:
        - saved model logs in logs/
        - loss graphs in graphs/
        """

        # Creating Dataloader
        if args.data in DATASETS_IMPLEMENTED.keys():
            train_dataset_transforms = TRAIN_TRANSFORMS[args.data]
            train_dataset = DATASETS_IMPLEMENTED[args.data](args,  # Change this
                                                            False,
                                                            transform=train_dataset_transforms)
            train_dataloader = DataLoader(train_dataset, **kwargs_train)
        else:
            raise RuntimeError(f'Dataset object not implemented for {args.data}')

        # Creating model
        if args.data in MODELS.keys():
            model = MODELS[args.data]
            init_mode = networks.init_weights(model, args.verbose, init_type='normal')
            if args.cuda:
                model = model.cuda()
        else:
            raise RuntimeError(f'Model object not implemented for {args.data}')

        # Useful variables
        nb_img_train = len(train_dataset)
        nb_classes = train_dataset.nb_classes
        percent_labeled = train_dataset.percent_labeled
        nb_batches = len(train_dataloader)

        # Creating training method
        if args.method in METHODS_IMPLEMENTED:
            method = METHODS_IMPLEMENTED[args.method](args.hyperparameters, nb_img_train, nb_classes, percent_labeled, nb_batches, args.batch_size, args.log_interval, args.cuda, args.verbose)
        else:
            raise RuntimeError(f'Method not implemented: {args.method}')

        # Creating optimizer
        if args.optimizer == 'Adam':
            optimizer = Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
        elif args.optimizer == 'SGD':
            optimizer = None  # TODO!
        else:
            raise RuntimeError(f'Optimizer not implemented: {args.optimizer}')

        train_info = utils.get_train_info(nb_img_train, nb_classes, percent_labeled, args.epochs, args.batch_size, nb_batches, args.shuffle, method, args.train_id, optimizer)

        if args.verbose:
            print(train_info)
            print(model)
            print(f'Initialisation mode: {init_mode}')
            print(utils.get_nb_parameters(model))

        with open(os.path.join(args.trained_model_path, 'info.txt'), 'a+') as f:
            f.write(train_info)

        method.train(train_dataloader, model, optimizer, nb_img_train, nb_classes, nb_batches, args.batch_size, args.epochs,
                     args.trained_model_path, args.verbose)  # Doesn't return anything, just saves all relevant data its dedicated folder

    def test(args):
        """
        Runs a set number of tests on a given trained model
        --------------------------------------
        Inputs:
        - data type and dataset_name
        - train id
        Outputs:
        - classification report and metrics in results.txt
        """

        # Creating Dataloader
        if args.data in DATASETS_IMPLEMENTED.keys():
            test_dataset_transforms = TEST_TRANSFORMS[args.data]
            test_dataset = DATASETS_IMPLEMENTED[args.data](args,  # Change this
                                                           True,
                                                           transform=test_dataset_transforms)
            test_dataloader = DataLoader(test_dataset, **kwargs_test)
        else:
            raise RuntimeError(f'Dataset object not implemented for {args.data}')

        # Loading model
        if args.data in MODELS.keys():
            model = MODELS[args.data]
            logs_path = os.path.join(args.trained_model_path, 'logs')
            latest_log = utils.get_latest_log(logs_path)
            checkpoint = torch.load(os.path.join(logs_path, latest_log))
            model.load_state_dict(checkpoint['state_dict'])
            if args.cuda:
                model = model.cuda()
        else:
            raise RuntimeError(f'Model object not implemented for {args.data}')

        # Creating testing class
        test_method = methods.TestingClass(args.verbose, args.cuda)

        report = test_method.test(test_dataloader, model, args.test_runs)
        report = f'Number of test runs: {args.test_runs}\n' + report

        with open(os.path.join(args.trained_model_path, 'results.txt'), 'w+') as f:
            f.write(report)

# ------------------------------------------------------------------------------

    if not args.test:
        args.train_id = utils.get_train_id()
        args.full_name = args.dataset_name + '_' + args.method + '_' + args.train_id

    else:
        if args.train_id != None:
            args.full_name = utils.get_trained_model_from_id(args.train_id)
        else:
            raise RuntimeError('Please provide a train_id in order to run tests')

    args.trained_model_path = os.path.join(TRAINED_MODELS_PATH, args.full_name)

    if args.train or args.test or args.params_optim:

        header = f'Data: {args.data}\n'
        header += f'Dataset name: {args.dataset_name}\n'
        header += f'Image mode: {args.img_mode}'

        print('')
        print(header)
        print('Cuda: ', args.cuda)
        print('-----------------------------------------------')

        header += '\n-----------------------------------------------\n'

        if not os.path.exists(args.trained_model_path):
            os.makedirs(args.trained_model_path)

        with open(os.path.join(args.trained_model_path, 'info.txt'), 'w+') as f:
            f.write(header)

    else:
        print('Please specify the task you want to run')

    if args.train:  # Trains one model

        print('Starting training...')

        args.hyperparameters = HYPERPARAMETERS_DEFAULT[args.method]
        train(args)

        print('Training done!')

    if args.test:  # Tests one trained model

        print('Running tests...')

        test(args)

        print('Tests done!')

    if args.params_optim:  # From a number of hyperparamters, returns the set that goves the best performance

        print('Hyperparamters optimization...')

        # Sombre invocation

        print('Search done!')


if __name__ == '__main__':
    main()