import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


class MNISTModel(nn.Module):

    def __init__(self, p=0.5, fm1=16, fm2=32):
        super(MNISTModel, self).__init__()
        self.fm1 = fm1
        self.fm2 = fm2
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p)
        self.conv1 = weight_norm(nn.Conv2d(1, self.fm1, 3, padding=1))
        self.conv2 = weight_norm(nn.Conv2d(self.fm1, self.fm2, 3, padding=1))
        self.mp = nn.MaxPool2d(3, stride=2, padding=1)
        self.fc = nn.Linear(self.fm2 * 7 * 7, 10)

    def forward(self, x):
        x = self.act(self.mp(self.conv1(x)))
        x = self.act(self.mp(self.conv2(x)))
        x = x.view(-1, self.fm2 * 7 * 7)
        x = self.drop(x)
        x = self.fc(x)
        return x

class CIFAR10Model(nn.Module):

    def __init__(self):
        super(CIFAR10Model, self).__init__()
        self.act = nn.LeakyReLU(negative_slope=0.1)
        self.pool = nn.MaxPool2d(2, 2)
        self.drop = nn.Dropout2d()
        self.avg_pool = nn.AvgPool2d(6)

        self.conv1_in = nn.Conv2d(3, 128, 3)
        self.conv1 = nn.Conv2d(128, 128, 3)
        self.conv2_in = nn.Conv2d(128, 256, 3)
        self.conv2 = nn.Conv2d(256, 256, 3)
        self.conv3_a = nn.Conv2d(256, 512, 3)
        self.conv3_b = nn.Conv2d(512, 256, 1)
        self.conv3_c = nn.Conv2d(256, 128, 1)
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        x = self.act(self.conv1_in(x))
        x = self.act(self.conv1(x))
        x = self.act(self.conv1(x))
        x = self.drop(self.pool(x))

        x = self.act(self.conv2_in(x))
        x = self.act(self.conv2(x))
        x = self.act(self.conv2(x))
        x = self.drop(self.pool(x))

        x = self.act(self.conv3_a(x))
        x = self.act(self.conv3_b(x))
        x = self.act(self.conv3_c(x))

        x = torch.squeeze(x)
        x = self.fc(x)

        return x
