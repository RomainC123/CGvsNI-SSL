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

    def __init__(self, **kwargs):

        super(AdamContainer, self).__init__(**kwargs)

        self.name = 'Adam'

        self.max_lr = kwargs['max_lr']
        self.beta1 = kwargs['beta1']
        self.beta2 = kwargs['beta2']

        self.lr_schedule = LR_SCHEDULE
        self.beta1_schedule = B1_SCHEDULE

    def create_optim(self, model):

        self.optim = Adam(model.model.parameters(), lr=0., betas=(self.beta1, self.beta2))

    def update_params(self, epoch, total_epochs):

        for param_group in self.optim.param_groups:
            param_group['lr'] = self.lr_schedule(epoch, total_epochs) * self.max_lr
            param_group['betas'] = self.beta1_schedule(epoch, total_epochs) * self.beta1, self.beta2

    def get_info(self):

        infos = super(AdamContainer, self).get_info()

        infos += f'Max learning rate: {self.max_lr}\n'
        infos += f'Betas: {(self.beta1, self.beta2)}\n'
        infos += 'Learning rate schedule:\n'
        infos += self.lr_schedule.get_info()
        infos += '\nBeta 1 schedule:\n'
        infos += self.beta1_schedule.get_info()

        return infos
