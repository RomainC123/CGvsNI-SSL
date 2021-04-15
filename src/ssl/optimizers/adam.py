################################################################################
#   Librairies                                                                 #
################################################################################

from torch.optim import Adam

from .base import BaseOptimizerContainer
from ..utils.schedules import LR_SCHEDULE, B1_SCHEDULE

################################################################################
#   Adam Optimizer class                                                       #
################################################################################


class AdamContainer(BaseOptimizerContainer):

    def __init__(self, hyperparameters):

        super(AdamContainer, self).__init__(hyperparameters)

        self.name = 'Adam'

        self.max_lr = hyperparameters['max_lr']
        self.beta1 = hyperparameters['beta1']
        self.beta2 = hyperparameters['beta2']

        self.lr_schedule = LR_SCHEDULE
        self.beta1_schedule = B1_SCHEDULE

    def __call__(self, model, epoch, total_epochs):

        lr = self.lr_schedule(epoch, total_epochs) * self.max_lr
        beta1 = 0.5 + self.beta1_schedule(epoch, total_epochs) * (self.beta1 - 0.5)
        beta2 = self.beta2

        return Adam(model.model.parameters(), lr=lr, betas=(beta1, beta2))

    def get_info(self):

        infos = super(AdamContainer, self).get_info()

        infos += f'Max learning rate: {self.max_lr}\n'
        infos += f'Betas: {(self.beta1, self.beta2)}\n'
        infos += 'Learning rate schedule:\n'
        infos += self.lr_schedule.get_info()
        infos += '\nBeta 1 schedule:\n'
        infos += self.beta1_schedule.get_info()

        return infos
