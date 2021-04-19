################################################################################
#   Libraries                                                                  #
################################################################################

import os
import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from distutils.dir_util import copy_tree
from datetime import datetime
from tqdm import tqdm

from ssl.utils.constants import BATCH_SIZE, DEFAULT_EPOCHS
from ssl.utils.paths import TRAINED_MODELS_PATH
from ssl.utils.functionalities import DATASETS, MODELS, OPTIMIZERS, METHODS
from ssl.utils.hyperparameters import METHODS_DEFAULT, OPTIMIZERS_DEFAULT

import warnings
warnings.filterwarnings('ignore')

################################################################################
#   Argparse                                                                   #
################################################################################


def get_args():

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

    # Training parameters
    parser.add_argument('--seed', type=int, default=0, help='seed used to generate the dataset')
    parser.add_argument('--data', type=str, help='data to use')
    parser.add_argument('--nb_samples_test', type=int, help='number of testing samples')
    parser.add_argument('--nb_samples_labeled', type=int, help='number of labeled samples in the training set')
    parser.add_argument('--img_mode', type=str, help='loading method (RGB or L)')
    parser.add_argument('--model', type=str, help='model to use')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer to use')
    parser.add_argument('--method', type=str, help='method to use')
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS, help='number of epochs to train (default: 300)')

    # Hardware parameter
    parser.add_argument('--no-verbose', dest='verbose', action='store_false')
    parser.set_defaults(verbose=True)
    parser.add_argument('--no-cuda', dest='no_cuda', action='store_true')
    parser.set_defaults(no_cuda=False)

    args = parser.parse_args()

    assert(args.data != None)
    assert(args.nb_samples_test != None)
    assert(args.nb_samples_labeled != None)
    assert(args.img_mode != None)
    assert(args.model != None)
    assert(args.method != None)
    assert(args.epochs != None)

    return args

################################################################################
#   Main                                                                       #
################################################################################


def main():

    # Grabbing the result of the argparse
    args = get_args()

    # Cuda variable
    cuda_state = not args.no_cuda and torch.cuda.is_available()
    if cuda_state:
        cudnn.benchmark = True
        kwargs_hardware = {'batch_size': BATCH_SIZE, 'shuffle': False, 'num_workers': 8, 'pin_memory': True}
    else:
        kwargs_hardware = {'batch_size': BATCH_SIZE, 'shuffle': False}

    # Setting random seed
    np.random.seed(args.seed)

    model_path = os.path.join(TRAINED_MODELS_PATH, f'{args.data}' + '_{date:%d-%m-%Y_%H:%M:%S}'.format(date=datetime.now()))
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # Building all containers
    dataset = DATASETS[args.data](args.data, 10000, 1000, img_mode=args.img_mode)
    model = MODELS[args.model](dataset.nb_classes, 'normal')
    optimizer = OPTIMIZERS[args.optimizer](OPTIMIZERS_DEFAULT[args.optimizer])
    method = METHODS[args.method](METHODS_DEFAULT[args.method])

    if cuda_state:
        model.cuda()
        method.cuda()

    # Saving all infos
    if args.verbose:
        print(dataset.get_info())
        print('------------------------------------\n' + model.get_info())
        print('------------------------------------\n' + optimizer.get_info())
        print('------------------------------------\n' + method.get_info())
    with open(os.path.join(model_path, 'info.txt'), 'a+') as f:
        f.write(dataset.get_info())
        f.write('\n------------------------------------\n' + model.get_info())
        f.write('\n------------------------------------\n' + optimizer.get_info())
        f.write('\n------------------------------------\n' + method.get_info())

    print('\nStarting training...')
    method.train(dataset, model, optimizer, 0, args.epochs, model_path, args.verbose)
    print('Training done\n')

    print('Testing...')
    method.test(dataset, model, model_path, args.verbose)
    print('Testing done')


if __name__ == '__main__':
    main()
