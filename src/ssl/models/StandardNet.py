import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from .base import BaseModelContainer


class GaussianNoise(nn.Module):
    def __init__(self, sigma):
        super(GaussianNoise, self).__init__()
        self.sigma = sigma

    def forward(self, input):
        if self.training:
            noise = Variable(input.data.new(input.size()).normal_(std=self.sigma))
            return input + noise
        else:
            return input


class StandardNet(nn.Module):
    def __init__(self, num_classes=10):
        super(StandardNet, self).__init__()

        self.gn = GaussianNoise(0.15)
        self.activation = nn.LeakyReLU(0.1)

        self.conv1a = nn.Conv2d(3, 128, 3, padding=1, bias=True)
        self.conv1b = nn.Conv2d(128, 128, 3, padding=1, bias=True)
        self.conv1c = nn.Conv2d(128, 128, 3, padding=1, bias=True)
        self.mp1 = nn.MaxPool2d(2, stride=2, padding=0)
        self.drop1 = nn.Dropout(0.5)

        self.conv2a = nn.Conv2d(128, 256, 3, padding=1, bias=True)
        self.conv2b = nn.Conv2d(256, 256, 3, padding=1, bias=True)
        self.conv2c = nn.Conv2d(256, 256, 3, padding=1, bias=True)
        self.mp2 = nn.MaxPool2d(2, stride=2, padding=0)
        self.drop2 = nn.Dropout(0.5)

        self.conv3a = nn.Conv2d(256, 512, 3, padding=0, bias=True)
        self.conv3b = nn.Conv2d(512, 256, 1, padding=0, bias=True)
        self.conv3c = nn.Conv2d(256, 128, 1, padding=0, bias=True)
        self.ap3 = nn.AvgPool2d(6, stride=2, padding=0)

        self.fc1 = nn.Linear(128, num_classes, bias=True)

    def forward(self, x, moving_average=True, init_mode=False):

        if self.training:
            x = self.gn(x)
        else:
            moving_average = False
        x = self.activation(self.conv1a(x))
        x = self.activation(self.conv1b(x))
        x = self.activation(self.conv1c(x))
        x = self.mp1(x)
        x = self.drop1(x)

        x = self.activation(self.conv2a(x))
        x = self.activation(self.conv2b(x))
        x = self.activation(self.conv2c(x))
        x = self.mp2(x)
        x = self.drop2(x)

        x = self.activation(self.conv3a(x))
        x = self.activation(self.conv3b(x))
        x = self.activation(self.conv3c(x))
        x = self.ap3(x)

        return self.fc1(x.view(-1, 128))

################################################################################
#   SimpleNet Model container                                                  #
################################################################################


class StandardNetContainer(BaseModelContainer):

    def __init__(self, nb_classes, init_mode, model_path):

        self.name = 'StandardNet'
        self.model = StandardNet(nb_classes)

        super(StandardNetContainer, self).__init__(nb_classes, init_mode, model_path)
