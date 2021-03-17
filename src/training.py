################################################################################
#   Libraries                                                                  #
################################################################################

import os
import pathlib
import argparse

import temporal_ensembling
import datasets
import models
import utils
from vars import *

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim import Adam


################################################################################
#   Argparse                                                                   #
################################################################################

parser = argparse.ArgumentParser(description='Semi-supervised MNIST training')

# Data to use
parser.add_argument('--data', type=str, help='data to use')
parser.add_argument('--dataset_name', type=str, help='name of the saved dataset to use')
parser.add_argument('--img_mode', type=str, help='loading method (RGB or L)')

# Training method and optimizer
parser.add_argument('--method', type=str, help='training method')
parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer to use')

# Training parameters
parser.add_argument('--batch_size', type=int, default=100, help='input batch size for training (default: 100)')
parser.add_argument('--shuffle', type=bool, default=True, help='shuffle bool for train dataset (default: True)')
parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train (default: 300)')

# Whether or not to train, show graphs and/or test
parser.add_argument('--graph', dest='graph', action='store_true')
parser.add_argument('--no-graph', dest='graph', action='store_false')
parser.set_defaults(graph=False)

# Hardware parameter
parser.add_argument('--log_interval', type=int, default=10, help='how many batches to wait before logging training status')
parser.add_argument('--no_cuda', default=False, help='disables CUDA training')

args = parser.parse_args()

args.TRAIN_STEP = TRAIN_STEP

args.logs_path = os.path.join(LOGS_PATH, args.data, args.dataset_name)
if not os.path.exists(args.logs_path):
    os.makedirs(args.logs_path)

################################################################################
#   Cuda                                                                       #
################################################################################

args.cuda = not args.no_cuda and torch.cuda.is_available()
if args.cuda:
    cudnn.benchmark = True

if args.cuda:
    kwargs = {'batch_size': args.batch_size, 'shuffle': args.shuffle, 'num_workers': 8, 'pin_memory': True}
else:
    kwargs = {'batch_size': args.batch_size, 'shuffle': args.shuffle}

################################################################################
#   Training                                                                   #
################################################################################


def main():

    def get_train_id(args):
        """
        Grabs the lowest availiable training id
        """

        list_train = [fname.split('_') for fname in os.listdir(args.logs_path)]
        list_idx_taken = []
        for method, idx in list_train:
            if method == args.method:
                list_idx_taken.append(int(idx))

        idx = 1
        while idx in list_idx_taken:
            idx += 1
        return idx

    print('Starting training...')

    # Getting training id, to create paths
    args.train_id = get_train_id(args)

    # Creating all relevant paths
    args.logs_path_full = os.path.join(args.logs_path, args.method + '_' + str(args.train_id))
    if not os.path.exists(args.logs_path_full):
        os.makedirs(args.logs_path_full)

    args.graphs_path_full = os.path.join(GRAPHS_PATH, args.data, args.dataset_name, args.method + '_' + str(args.train_id))
    if not os.path.exists(args.graphs_path_full):
        os.makedirs(args.graphs_path_full)

    # Creating Dataset object
    if args.data in DATASETS_IMPLEMENTED.keys():
        train_dataset_transforms = TRANSFORMS_TRAIN[args.data]
        train_dataset = DATASETS_IMPLEMENTED[args.data](args,
                                                        False,
                                                        transform=train_dataset_transforms)
        model = MODELS[args.data]
    else:
        raise RuntimeError('Data type not implemented')

    print('Image mode: ', args.img_mode)

    # DataLoader object
    train_dataloader = DataLoader(train_dataset, **kwargs)

    # Useful variables
    args.nb_img_train = len(train_dataset)
    args.nb_batches = len(train_dataloader)
    args.nb_classes = train_dataset.nb_classes
    args.percent_labeled = train_dataset.percent_labeled

    print('Number of train data: {}'.format(len(train_dataloader.dataset)))

    # Optimizer
    if args.optimizer == 'Adam':
        optimizer = Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
    if args.optimizer == 'SGD':
        optimizer = None  # TODO!

    # Choosing training method
    if args.method == 'TemporalEnsembling':
        # Adding all hyperparameters to args
        args.alpha = HYPERPARAMETERS[args.method]['alpha']
        args.ramp_epochs = HYPERPARAMETERS[args.method]['ramp_epochs']
        args.ramp_mult = HYPERPARAMETERS[args.method]['ramp_mult']
        args.max_weight = HYPERPARAMETERS[args.method]['max_weight']
        temporal_ensembling.training(train_dataloader, model, optimizer, args)

    print('Training done!')

    # Displaying the training report
    if args.graph:
        utils.training_report(args)


if __name__ == '__main__':
    main()
