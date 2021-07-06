################################################################################
#   Libraries                                                                  #
################################################################################

import os
import argparse
import numpy as np
import torch

from datetime import datetime

from ssl.utils.constants import BATCH_SIZE, DEFAULT_EPOCHS, CGVSNI_DATASETS_IDS
from ssl.utils.paths import TRAINED_MODELS_PATH
from ssl.utils.datasets import DATASETS
from ssl.utils.models import MODELS
from ssl.utils.optimizers import OPTIMIZERS
from ssl.utils.methods import METHODS
from ssl.utils.hyperparameters import METHODS_DEFAULT, OPTIMIZERS_DEFAULT
from ssl.utils.tools import save_info

import warnings
warnings.filterwarnings('ignore')

################################################################################
#   Argparse                                                                   #
################################################################################


def get_args():

    parser = argparse.ArgumentParser(description='CGvsNI testing script')

    # Training parameters
    parser.add_argument('--seed', type=int, default=0, help='seed used to generate the dataset')
    parser.add_argument('--folder', type=str, default=None, help='subfolder in which to save the model (default: None)')
    parser.add_argument('--name', type=str, default=None, help='name of the saved model')
    parser.add_argument('--day', type=str, help='day of the trained model to load')
    parser.add_argument('--hour', type=str, help='hour of the trained model')

    parser.add_argument('--datasets_to_use', type=str, help='datasets to use for training')
    parser.add_argument('--label_mode', type=str, help='either biclass or multiclass')
    parser.add_argument('--img_mode', type=str, help='loading method (RGB or L)')

    parser.add_argument('--nb_samples_train', type=int, default=10800, help='number of testing samples')
    parser.add_argument('--nb_samples_test', type=int, default=720, help='number of testing samples')
    parser.add_argument('--nb_samples_labeled', type=int, default=1080, help='number of labeled samples in the training set')

    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer to use')
    parser.add_argument('--max_lr', type=float, default=0.0002, help='max learning rate to use, if applicable')

    parser.add_argument('--method', type=str, help='method to use')
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS, help='number of epochs to train (default: 300)')

    # Hardware parameter
    parser.add_argument('--no-verbose', dest='verbose', action='store_false')
    parser.set_defaults(verbose=True)
    parser.add_argument('--no-cuda', dest='no_cuda', action='store_true')
    parser.set_defaults(no_cuda=False)

    args = parser.parse_args()

    assert(args.img_mode != None)
    assert(args.method != None)
    assert(args.epochs != None)

    return args

################################################################################
#   Main                                                                       #
################################################################################


def main():

    # Grabbing the result of the argparse
    args = get_args()
    args.data = 'CGvsNI'
    args.init_mode = 'Nope'

    # Cuda variable
    cuda_state = not args.no_cuda and torch.cuda.is_available()

    # Setting random seed
    np.random.seed(args.seed)

    if args.folder != None:
        main_folder_path = os.path.join(TRAINED_MODELS_PATH, args.folder)
    else:
        main_folder_path = TRAINED_MODELS_PATH

    if args.name != None:
        model_path = os.path.join(main_folder_path, f'{args.name}' + '_{date:%d-%m-%Y_%H:%M:%S}'.format(date=datetime.now()))
    elif args.datasets_to_use != None:
        model_path = os.path.join(main_folder_path, f'{args.data}_{args.datasets_to_use}_{args.method}' + '_{date:%d-%m-%Y_%H:%M:%S}'.format(date=datetime.now()))
    else:
        model_path = os.path.join(main_folder_path, f'{args.data}_{args.method}' + '_{date:%d-%m-%Y_%H:%M:%S}'.format(date=datetime.now()))

    # Sheitan mais ça marche donc menfou
    if args.method == 'FullSup':
        args.nb_samples_labeled = args.nb_samples_train

    dataset_train = DATASETS[args.data](args.data, args.nb_samples_train, args.nb_samples_test, args.nb_samples_labeled, cuda_state, img_mode=args.img_mode, datasets_to_use=args.datasets_to_use, label_mode=args.label_mode, epsilon=1e-1)
    model = MODELS['ENet'](dataset_train.nb_classes, args.init_mode, model_path)
    optimizer = OPTIMIZERS[args.optimizer](max_lr=args.max_lr, **OPTIMIZERS_DEFAULT[args.optimizer])
    method = METHODS[args.method](model=model, percent_labeled=dataset_train.percent_labeled, **METHODS_DEFAULT[args.method])

    if cuda_state:
        model.cuda()
        method.cuda()

    print('\nStarting training...')
    save_info(model_path, dataset_train, model, optimizer, method, args.verbose)
    method.train(dataset_train, model, optimizer, 0, args.epochs, model_path, args.verbose)
    print('Training done\n')

    for dataset_name in CGVSNI_DATASETS_IDS.keys():
        print(f'Testing for {dataset_name}...')
        dataset_test = DATASETS[args.data](args.data, args.nb_samples_train, args.nb_samples_test, args.nb_samples_labeled, cuda_state, img_mode=args.img_mode, datasets_to_use=dataset_name, label_mode=args.label_mode, epsilon=1e-1)
        method.test(dataset_test, model, model_path, args.verbose, name=f'results_{dataset_name}.txt')
    print('Testing done')


if __name__ == '__main__':
    main()
