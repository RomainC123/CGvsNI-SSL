################################################################################
#   Libraries                                                                  #
################################################################################

import os
import pathlib
import argparse

import training
import testing
import models
import datasets
import display

import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

from torch.optim import Adam
from torch.utils.data import DataLoader


################################################################################
#   Paths and variables                                                        #
################################################################################

TRAIN_STEP = 10  # To be set manually

DATASETS_IMPLEMENTED = {
    'MNIST': datasets.DatasetMNIST,
    'CIFAR10': datasets.DatasetCIFAR10
}

HYPERPARAMETERS = {
    'TemporalEnsembling': {
        'alpha': 0.6,
        'ramp_epochs': 80,
        'ramp_mult': 5,
        'max_weight': 30.
    }
}


def get_dataset_caracs(dataset, dataset_name):

    dataset_train = dataset(args, False)
    dataset_test = dataset(args, True)
    nb_train, percent_labeled = len(dataset_train), dataset_train.percent_labeled
    nb_test = len(dataset_test)

    return nb_train, nb_test, percent_labeled


ROOT_PATH = pathlib.Path(__file__).resolve().parents[1].absolute()

DATASETS_PATH = os.path.join(ROOT_PATH, 'datasets')  # dataset.csv files path
if not os.path.exists(DATASETS_PATH):
    os.makedirs(DATASETS_PATH)

GRAPHS_PATH = os.path.join(ROOT_PATH, 'graphs')  # Graphs path
if not os.path.exists(GRAPHS_PATH):
    os.makedirs(GRAPHS_PATH)

LOGS_PATH = os.path.join(ROOT_PATH, 'logs')  # Logs path
if not os.path.exists(LOGS_PATH):
    os.makedirs(LOGS_PATH)

TEST_RESULTS_PATH = os.path.join(ROOT_PATH, 'test_results')
if not os.path.exists(TEST_RESULTS_PATH):
    os.makedirs(TEST_RESULTS_PATH)

################################################################################
#   Argparse                                                                   #
################################################################################

parser = argparse.ArgumentParser(description='Semi-supervised MNIST training')

# Data to use
parser.add_argument('--data', type=str, help='data to use')
parser.add_argument('--dataset_name', type=str, help='name of the saved dataset to use')
parser.add_argument('--img_mode', type=str, help='loading method (RGB or L)')
parser.add_argument('--train_id', type=int, help='index of the trained model to load for tests. In case of training, gets overwritten')

# Training method and optimizer
parser.add_argument('--method', type=str, help='training method')
parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer to use')

# Training parameters
parser.add_argument('--batch_size', type=int, default=100, help='input batch size for training (default: 100)')
parser.add_argument('--test_batch_size', type=int, default=50, help='input batch size for testing (default: 50)')
parser.add_argument('--shuffle', type=bool, default=True, help='shuffle bool for train dataset (default: True)')
parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train (default: 300)')

# # Method hyperparameters
# parser.add_argument('--alpha', type=float, default=0.6, help='variable for the moving average part (default: 0.6)')
# parser.add_argument('--ramp_epochs', type=int, default=80, help='number of epochs before unsupervised weight reaches its maximum (default: 50)')
# parser.add_argument('--ramp_mult', type=int, default=5, help='multiplier to be used in the exponential for the calculation of the weight of the unsupervised part of the loss (default: 4)')
# parser.add_argument('--max_weight', type=float, default=30., help='maximum weight for the unsupervised loss (default: 30.)')

# Whether or not to train, show graphs and/or test
parser.add_argument('--train', dest='train', action='store_true')
parser.add_argument('--no-train', dest='train', action='store_false')
parser.set_defaults(train=True)
parser.add_argument('--graph', dest='graph', action='store_true')
parser.add_argument('--no-graph', dest='graph', action='store_false')
parser.set_defaults(graph=False)
parser.add_argument('--test', dest='test', action='store_true')
parser.add_argument('--no-test', dest='test', action='store_false')
parser.set_defaults(test=True)

# Hardware parameter
parser.add_argument('--log_interval', type=int, default=10, help='how many batches to wait before logging training status')
parser.add_argument('--no_cuda', default=False, help='disables CUDA training')

args = parser.parse_args()

# New args
args.TRAIN_STEP = TRAIN_STEP

args.graphs_path = os.path.join(GRAPHS_PATH, args.data, args.dataset_name)
if not os.path.exists(args.graphs_path):
    os.makedirs(args.graphs_path)

args.logs_path = os.path.join(LOGS_PATH, args.data, args.dataset_name)
if not os.path.exists(args.logs_path):
    os.makedirs(args.logs_path)

args.results_path = os.path.join(TEST_RESULTS_PATH, args.data, args.dataset_name)
if not os.path.exists(args.results_path):
    os.makedirs(args.results_path)

args.nb_train, args.nb_test, args.percent_labeled = get_dataset_caracs(DATASETS_IMPLEMENTED[args.data], args)

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
#   Main                                                                       #
################################################################################


def main():

    if args.train:

        print('Starting training...')

        args.train_id = get_train_id(args)
        print(args.train_id)

        args.graphs_path_full = os.path.join(GRAPHS_PATH, args.data, args.dataset_name, args.method + '_'  + str(args.train_id))
        if not os.path.exists(args.graphs_path_full):
            os.makedirs(args.graphs_path_full)

        args.logs_path_full = os.path.join(LOGS_PATH, args.data, args.dataset_name, args.method + '_'  + str(args.train_id))
        if not os.path.exists(args.logs_path_full):
            os.makedirs(args.logs_path_full)

        # Dataset object based on which data is used
        if args.data == 'MNIST':
            train_dataset_transforms = transforms.Compose([transforms.ToTensor(),
                                                           transforms.Normalize((0.5), (0.5))])
            train_dataset = datasets.DatasetMNIST(args,
                                                  False,
                                                  transform=train_dataset_transforms)
            model = models.MNISTModel()

        if args.data == 'CIFAR10':
            train_dataset_transforms = transforms.Compose([transforms.ToTensor(),
                                                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            train_dataset = datasets.DatasetCIFAR10(args,
                                                    False,
                                                    transform=train_dataset_transforms)
            model = models.CIFAR10Model()

        if args.data == 'CGvsNI':  # TODO!
            train_dataset_transforms = None

        # DataLoader object
        print('Image mode: ', args.img_mode)
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

        # Training method
        if args.method == 'TemporalEnsembling':
            args.alpha = HYPERPARAMETERS[args.method]['alpha']
            args.ramp_epochs = HYPERPARAMETERS[args.method]['ramp_epochs']
            args.ramp_mult = HYPERPARAMETERS[args.method]['ramp_mult']
            args.max_weight = HYPERPARAMETERS[args.method]['max_weight']
            training.temporal_ensembling_training(train_dataloader, model, optimizer, args)

        print('Training done!')

    if args.graph:
        display.show_loss(args)

    if args.test:
        args.logs_path_full = os.path.join(LOGS_PATH, args.data, args.dataset_name, args.method + '_'  + str(args.train_id))
        if not os.path.exists(args.logs_path_full):
            raise RuntimeError('No model chekpoint found, please train a model')

        print('Running tests...')

        if args.data == 'MNIST':
            test_dataset_transforms = transforms.Compose([transforms.ToTensor(),
                                                          transforms.Normalize((0.5), (0.5))])
            test_dataset = datasets.DatasetMNIST(args,
                                                 True,
                                                 transform=test_dataset_transforms)
            model = models.MNISTModel()

            latest_log = get_latest_log(args.logs_path_full)

            checkpoint = torch.load(os.path.join(args.logs_path_full, latest_log))
            model.load_state_dict(checkpoint['state_dict'])

        if args.data == 'CIFAR10':
            test_dataset_transforms = transforms.Compose([transforms.ToTensor(),
                                                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            test_dataset = datasets.DatasetCIFAR10(args,
                                                   True,
                                                   transform=test_dataset_transforms)
            model = models.CIFAR10Model()

            latest_log = get_latest_log(args.logs_path_full)

            checkpoint = torch.load(os.path.join(args.logs_path_full, latest_log))
            model.load_state_dict(checkpoint['state_dict'])

        if args.data == 'CGvsNI':  # TODO!
            test_dataset_transforms = None

        # DataLoader object
        test_dataloader = DataLoader(test_dataset, **kwargs)

        args.nb_img_test = len(test_dataset)
        args.nb_batches_test = len(test_dataloader)

        testing.testing_metrics(test_dataloader, model, args)
        testing.testing_display(test_dataloader, model, args)

        print('Tests done!')

################################################################################
#   Misc functions                                                             #
################################################################################


def get_train_id(args):

    list_train = [fname.split('_') for fname in os.listdir(args.logs_path)]
    list_idx_taken = []
    for method, idx in list_train:
        if method == args.method:
            list_idx_taken.append(int(idx))

    idx = 1
    while idx in list_idx_taken:
        idx += 1
    return idx


def get_latest_log(logs_path):

    list_logs = os.listdir(logs_path)

    latest_log_id = 0
    latest_log_epoch = 0
    latest_log = list_logs[0]

    for i in range(len(list_logs)):
        log_epoch = list_logs[i].split('_')[-1].split('.')[0]
        if int(log_epoch) > latest_log_epoch:
            latest_log = list_logs[i]
            latest_log_epoch = int(log_epoch)
            latest_log_id = i

    return latest_log


if __name__ == '__main__':
    main()
