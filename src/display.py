################################################################################
#   Libraries                                                                  #
################################################################################

import argparse
import random
import pickle
import numpy as np
from vars import *

import methods
import datasets
import models
import utils

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

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

NB_IMGS_TO_SHOW = 9

################################################################################
#   Argparse                                                                   #
################################################################################

parser = argparse.ArgumentParser(description='Semi-supervised MNIST display')

# Functionalities
parser.add_argument('--graphs', dest='graphs', action='store_true')
parser.set_defaults(graphs=False)
parser.add_argument('--examples', dest='examples', action='store_true')
parser.set_defaults(examples=False)

# Data to use
parser.add_argument('--data', type=str, help='data to use')
parser.add_argument('--dataset_name', type=str, help='name of the saved dataset to use')
parser.add_argument('--img_mode', type=str, help='loading method (RGB or L)')
parser.add_argument('--train_id', type=int, help='index of the trained model to load for tests. In case of training, gets overwritten')

# Hardware parameter
parser.add_argument('--no_cuda', default=False, help='disables CUDA')

args = parser.parse_args()

if args.img_mode == None:
    raise RuntimeError('Please specify img_mode param')

################################################################################
#   Cuda                                                                       #
################################################################################

args.cuda = not args.no_cuda and torch.cuda.is_available()

################################################################################
#   Displays                                                                   #
################################################################################


def main():

    def show_graphs(args):

        graphs_path = os.path.join(args.trained_model_path, 'graphs')

        with open(os.path.join(graphs_path, 'loss.pkl'), 'rb') as f:
            losses = pickle.load(f)

        with open(os.path.join(graphs_path, 'sup_loss.pkl'), 'rb') as f:
            sup_losses = pickle.load(f)

        with open(os.path.join(graphs_path, 'unsup_loss.pkl'), 'rb') as f:
            unsup_losses = pickle.load(f)

        epochs = len(losses)

        fig = plt.figure(figsize=(12, 24))

        ax1 = fig.add_subplot(311)
        ax1.set_title('Loss')
        ax1.plot(range(epochs), losses)

        ax2 = fig.add_subplot(312)
        ax2.set_title('Supervised Loss')
        ax2.plot(range(epochs), sup_losses)

        ax3 = fig.add_subplot(313)
        ax3.set_title('Unsupervised Loss')
        ax3.plot(range(epochs), unsup_losses)

        plt.show()

    def show_example(args):

        if args.data in DATASETS_IMPLEMENTED.keys():
            test_dataset_transforms = TEST_TRANSFORMS[args.data]
            test_dataset = DATASETS_IMPLEMENTED[args.data](args,
                                                           'default',
                                                           True,
                                                           transform=test_dataset_transforms)
            model = MODELS[args.data]
            logs_path = os.path.join(args.trained_model_path, 'logs')
            latest_log, epoch = utils.get_latest_log(logs_path)
            checkpoint = torch.load(os.path.join(logs_path, latest_log))
            model.load_state_dict(checkpoint['state_dict'])
        else:
            raise RuntimeError('Data type not implemented')

        if args.cuda:
            model = model.cuda()

        nb_imgs_to_show = NB_IMGS_TO_SHOW

        fig = plt.figure(figsize=(12, 12))

        id_to_show = random.sample(range(len(test_dataset)), nb_imgs_to_show)
        subplot_id = 1

        for i in id_to_show:

            img, target = test_dataset[i]
            if args.cuda:
                img = img.cuda()

            result = model(torch.unsqueeze(img, 0))
            result = F.softmax(result, dim=1)
            pred_label = result.data.max(1, keepdim=True)[1]

            img = img.cpu().numpy()
            if img.shape[1] == img.shape[2]:
                img = np.transpose(img, (1, 2, 0))  # Edge case can be annoying
            if np.amin(img) < 0:
                img = ((img / 2 + 0.5) * 255).astype(np.uint8)

            ax = fig.add_subplot(3, 3, subplot_id)
            if args.img_mode == 'L':
                ax.imshow(img, cmap='gray_r')
            elif args.img_mode == 'RGB':
                ax.imshow(img)
            ax.set_title(f'Prediction/True label: {pred_label.squeeze().cpu().numpy()}/{target}')
            ax.axis('off')

            subplot_id += 1

        plt.show()

# ------------------------------------------------------------------------------

    args.full_name = utils.get_trained_model_from_id(args.train_id)

    args.trained_model_path = os.path.join(TRAINED_MODELS_PATH, args.full_name)
    if not os.path.exists(args.trained_model_path):
        raise RuntimeError('Please provide a valid train_id')

    if args.graphs:

        show_graphs(args)

    if args.examples:

        show_example(args)


if __name__ == '__main__':
    main()
