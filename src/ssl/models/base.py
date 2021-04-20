import os
import torch

from ..utils.networks import init_weights
from ..utils.tools import get_latest_log

################################################################################
#   Base Model container                                                        #
################################################################################


class BaseModelContainer:
    """
    Base model container
    Given the model name, creates a wrapper with get_info, save and load functions
    """

    def __init__(self, nb_classes, init_mode):

        self.init_mode = init_mode
        if init_mode != 'pretrained':
            init_weights(self.model, self.init_mode)
        else:
            self.load()

    def _get_nb_parameters(self):

        nb_params = 0
        for param in self.model.parameters():
            nb_params += param.numel()

        return nb_params

    def cuda(self):
        self.model.cuda()

    def load(self, pretrained_path):
        latest_log, _ = get_latest_log(pretrained_path)
        checkpoint = torch.load(os.path.join(pretrained_path, 'logs', latest_log))
        self.model.load_state_dict(checkpoint['state_dict'])

    def save(self, path, epoch):
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save({'epoch': epoch,
                    'state_dict': self.model.state_dict()},
                   os.path.join(path, f'checkpoint_{epoch}.pth'))

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def test(self):
        self.model.test()

    def forward(self, x):
        return self.model.forward(x)

    def get_info(self):

        infos = f'Model: {self.name}\n'
        infos += str(self.model)
        infos += f'\nInit mode: {self.init_mode.capitalize()}\n'
        infos += f'Number of parameters: {self._get_nb_parameters()}'

        return infos
