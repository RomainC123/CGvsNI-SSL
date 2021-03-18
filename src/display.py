################################################################################
#   Libraries                                                                  #
################################################################################

import argparse
import random
import numpy as np

import utils
from vars import *

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

################################################################################
#   Argparse                                                                   #
################################################################################

parser = argparse.ArgumentParser(description='Semi-supervised MNIST display')

# Data to use
parser.add_argument('--data', type=str, help='data to use')
parser.add_argument('--dataset_name', type=str, help='name of the saved dataset to use')
parser.add_argument('--img_mode', type=str, help='loading method (RGB or L)')
parser.add_argument('--method', type=str, help='type of training used')
parser.add_argument('--train_id', type=int, help='index of the trained model to load for tests. In case of training, gets overwritten')

# Whether or not to show training report and/or classification examples
parser.add_argument('--graph', dest='graph', action='store_true')
parser.add_argument('--no-graph', dest='graph', action='store_false')
parser.set_defaults(graph=False)
parser.add_argument('--example', dest='example', action='store_true')
parser.add_argument('--no-example', dest='example', action='store_false')
parser.set_defaults(example=True)

# Hardware parameter
parser.add_argument('--no_cuda', default=False, help='disables CUDA training')

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

    if args.graph:
        training_report(args)

    if args.example:
        args.logs_path_full = os.path.join(LOGS_PATH, args.data, args.dataset_name, args.method + '_' + str(args.train_id))
        if not os.path.exists(args.logs_path_full):
            raise RuntimeError('Trained model not found, please check the given id')

        if args.data in DATASETS_IMPLEMENTED.keys():
            test_dataset_transforms = TRANSFORMS_TEST[args.data]
            test_dataset = DATASETS_IMPLEMENTED[args.data](args,
                                                           True,
                                                           transform=test_dataset_transforms)
            model = MODELS[args.data]
            latest_log = utils.get_latest_log(args.logs_path_full)
            checkpoint = torch.load(os.path.join(args.logs_path_full, latest_log))
            model.load_state_dict(checkpoint['state_dict'])
        else:
            raise RuntimeError('Data type not implemented')

        classification_display(test_dataset, model, args)


def training_report(args):

    with open(os.path.join(args.graphs_path, 'loss.pkl'), 'rb') as f:
        losses = pickle.load(f)

    with open(os.path.join(args.graphs_path, 'sup_loss.pkl'), 'rb') as f:
        sup_losses = pickle.load(f)

    with open(os.path.join(args.graphs_path, 'unsup_loss.pkl'), 'rb') as f:
        unsup_losses = pickle.load(f)

    fig = plt.figure(figsize=(12, 24))

    ax1 = fig.add_subplot(311)
    ax1.set_title('Loss')
    ax1.plot(range(args.epochs), losses)

    ax2 = fig.add_subplot(312)
    ax2.set_title('Supervised Loss')
    ax2.plot(range(args.epochs), sup_losses)

    ax3 = fig.add_subplot(313)
    ax3.set_title('Unsupervised Loss')
    ax3.plot(range(args.epochs), unsup_losses)

    plt.show()


def classification_display(test_dataset, model, args):

    if args.cuda:
        model = model.cuda()

    nb_imgs_to_show = 9
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


if __name__ == '__main__':
    main()
