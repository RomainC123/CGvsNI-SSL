import os
import torch

from ..utils.networks import init_weights

################################################################################
#   Base Model container                                                        #
################################################################################


class BaseModelContainer:
    """
    Base model container
    Given the model name, creates a wrapper with get_info, save and load functions
    """

    def __init__(self, init_mode):

        self.init_mode = init_mode
        init_weights(self.model, self.init_mode)

    def __call__(self):
        return self.model

    def _get_nb_parameters(self):

        nb_params = 0
        for param in self.model.parameters():
            nb_params += param.numel()

        return nb_params

    def load(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.init_mode = 'pretrained'

    def save(self, path, epoch):
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save({'epoch': epoch,
                    'state_dict': self.model.state_dict()},
                   os.path.join(path, f'checkpoint_{epoch}.pth'))

    def get_info(self):

        infos = f'Model: {self.name}\n'
        infos += str(self.model)
        infos += f'\nInit mode: {self.init_mode.capitalize()}\n'
        infos += f'Number of parameters: {self._get_nb_parameters()}'

        return infos
