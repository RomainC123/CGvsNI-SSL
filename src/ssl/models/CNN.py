import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseModelContainer
from ..utils.constants import STD

################################################################################
#   MNIST Model class                                                          #
################################################################################


class CNN(nn.Module):

    def __init__(self, nb_classes):
        super(CNN, self).__init__()
        self.gn = GaussianNoise(std=STD)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(0.5)
        self.conv1 = nn.utils.weight_norm(nn.Conv2d(1, 16, 3, padding=1))
        self.conv2 = nn.utils.weight_norm(nn.Conv2d(16, 32, 3, padding=1))
        self.mp = nn.MaxPool2d(3, stride=2, padding=1)
        self.fc = nn.Linear(32 * 7 * 7, nb_classes)

    def forward(self, x):
        if self.training:
            x = self.gn(x)
        x = self.act(self.mp(self.conv1(x)))
        x = self.act(self.mp(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = self.drop(x)
        x = self.fc(x)
        return x


class GaussianNoise(nn.Module):

    def __init__(self, shape=(100, 1, 28, 28), std=0.05):
        super(GaussianNoise, self).__init__()
        self.noise = torch.autograd.Variable(torch.zeros(shape).cuda())
        self.std = std

    def forward(self, x):
        c = x.shape[0]
        self.noise.data.normal_(0, std=self.std)
        return x + self.noise[:c]


################################################################################
#   MNIST Model container                                                      #
################################################################################


class CNNContainer(BaseModelContainer):

    def __init__(self, nb_classes, init_mode):

        self.name = 'CNN'
        self.model = CNN(nb_classes)

        super(CNNContainer, self).__init__(nb_classes, init_mode)
