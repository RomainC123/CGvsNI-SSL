import os
import pathlib
import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import dataloader_class

TRAIN_STEP = 100  # used for snapshot, and adjust learning rate
dataset_name = 'Artlantis_Autodesk_4000_0.5_0.2_0.1_data_512crop.csv'  # To be added to argparse
batch_size = 10

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
srm_trainable = False  # SRM trainable or not

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Classifier CGI vs NI')

parser.add_argument('--dataset_name', type=str,
                    help='name of the saved dataset to use')
parser.add_argument('--input_nc', type=int, default=3,
                    help='# of input image channels')
parser.add_argument('--img_mode', type=str, default='RGB',
                    help='chooses how image are loaded')

parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--test-batch-size', type=int, default=10, metavar='N',
                    help='input batch size for testing (default: 10)')
parser.add_argument('--epochs', type=int, default=TRAIN_STEP * 3, metavar='N',
                    help='number of epochs to train (default: 300)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--optimizer', default='sgd', type=str,
                    metavar='OPT', help='The optimizer to use (default: sgd)')

parser.add_argument('--log-dir', default='/logs',
                    help='folder to output model checkpoints')
parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')

args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()
if args.cuda:
    cudnn.benchmark = True
kwargs = {'num_workers': 8, 'pin_memory': True} if args.cuda else {}

ROOT_PATH = os.path.join(pathlib.Path(__file__).parent.absolute())
INPUT_PATH = os.path.join(ROOT_PATH, 'datasets', 'clean')

LOG_DIR = ROOT_PATH + args.log_dir
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

# dataset_path = os.path.join(INPUT_PATH, dataset_name)
train_loader = dataloader_class.DataLoaderCGNI(
    dataloader_class.ImageCGNIDataset(args, 'train',
                                      transforms.Compose([
                                          transforms.TenCrop(233),
                                          transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),  # returns a 4D tensor
                                          transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops]))
                                      ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)

print('The number of train data: {}'.format(len(train_loader.dataset)))


def main():
    """
    Define model, initialize all weights, create criterion and create_optimizer
    Then run train for each epoch
    """
    def train():
        """
        One epoch is one run of train
        Just the classic run of things
        Ruse the code here, and create new model and criterion
        """
