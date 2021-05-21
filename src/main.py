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
from ssl.utils.tools import save_info

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
    parser.add_argument('--test-lr', dest='test_lr', action='store_true')
    parser.set_defaults(test_lr=False)

    # Training parameters
    parser.add_argument('--seed', type=int, default=0, help='seed used to generate the dataset')
    parser.add_argument('--data', type=str, help='data to use')
    parser.add_argument('--datasets_to_use', type=str, help='datasets to use for training')
    parser.add_argument('--label_mode', type=str, help='either biclass or multiclass')
    parser.add_argument('--nb_samples_total', type=int, default=-1, help='number of testing samples')
    parser.add_argument('--nb_samples_test', type=int, default=10000, help='number of testing samples')
    parser.add_argument('--nb_samples_labeled', type=int, default=1000, help='number of labeled samples in the training set')
    parser.add_argument('--day', type=str, help='day of the trained model to load')
    parser.add_argument('--hour', type=str, help='hour of the trained model')
    parser.add_argument('--img_mode', type=str, help='loading method (RGB or L)')
    parser.add_argument('--model', type=str, help='model to use')
    parser.add_argument('--init_mode', type=str, default='normal', help='init mode to use')
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

    if args.day != None and args.hour != None:
        model_path = os.path.join(TRAINED_MODELS_PATH, f'{args.data}_{args.day}_{args.hour}')
    else:
        model_path = os.path.join(TRAINED_MODELS_PATH, f'{args.data}' + '_{date:%d-%m-%Y_%H:%M:%S}'.format(date=datetime.now()))

    # Building all containers
    dataset = DATASETS[args.data](args.data, args.nb_samples_total, args.nb_samples_test, args.nb_samples_labeled, cuda_state, img_mode=args.img_mode, datasets_to_use=args.datasets_to_use, label_mode=args.label_mode, epsilon=1e-1)
    model = MODELS[args.model](dataset.nb_classes, args.init_mode)
    optimizer = OPTIMIZERS[args.optimizer](**OPTIMIZERS_DEFAULT[args.optimizer])
    method = METHODS[args.method](**METHODS_DEFAULT[args.method])

    if cuda_state:
        model.cuda()
        method.cuda()

    if args.test_lr:
        base_model_path = os.path.join(TRAINED_MODELS_PATH, 'test_lr_' + f'{args.data}' + '_{date:%d-%m-%Y_%H:%M:%S}'.format(date=datetime.now()))
        print('Trying out a bunch of learning rates for only sup training...')
        lr_to_test = [0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001, 0.00003, 0.00001]
        for lr in lr_to_test:
            print(f'Testing lr={lr}...')
            model_path = os.path.join(base_model_path, str(lr))
            optimizer = OPTIMIZERS[args.optimizer](max_lr=lr, beta1=0.9, beta2=0.999)
            method = METHODS[args.method](alpha=0.6, max_unsup_weight=0.)
            if cuda_state:
                method.cuda()
            save_info(model_path, dataset, model, optimizer, method, args.verbose)
            method.train(dataset, model, optimizer, 0, args.epochs, model_path, args.verbose)
            method.test(dataset, model, model_path, args.verbose)
        print('Done!')

    if args.train_test:
        print('\nStarting training...')
        save_info(model_path, dataset, model, optimizer, method, args.verbose)
        method.train(dataset, model, optimizer, 0, args.epochs, model_path, args.verbose)
        print('Training done\n')

        print('Testing...')
        save_info(model_path, dataset, model, optimizer, method, args.verbose)
        method.test(dataset, model, model_path, args.verbose)
        print('Testing done')

    if args.train:
        print('\nStarting training...')
        save_info(model_path, dataset, model, optimizer, method, args.verbose)
        method.train(dataset, model, optimizer, 0, args.epochs, model_path, args.verbose)
        print('Training done')

    if args.test:
        print('Testing...')
        save_info(model_path, dataset, model, optimizer, method, args.verbose)
        method.test(dataset, model, model_path, args.verbose)
        print('Testing done')

if __name__ == '__main__':
    main()
