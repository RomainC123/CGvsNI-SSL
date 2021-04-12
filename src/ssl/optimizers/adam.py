################################################################################
#   Librairies                                                                 #
################################################################################

from torch.optim import Adam

from .base import BaseOptimizer
from ..utils.schedules import LR_SCHEDULE, B1_SCHEDULE

################################################################################
#   Adam Optimizer class                                                       #
################################################################################


class AdamOptimizer(BaseOptimizer):

    def __init__(self, **kwargs):

        super(AdamOptimizer, self).__init__(**kwargs)

        self.name = 'Adam'

        self.max_lr = kwargs['max_lr']
        self.beta1 = kwargs['beta1']
        self.beta2 = kwargs['beta2']

    def __call__(self, model, epoch, total_epochs):

        lr = LR_SCHEDULE(epoch, total_epochs) * self.max_lr
        beta1 = B1_SCHEDULE(epoch, total_epochs) * self.beta1
        beta2 = self.beta2

        return Adam(model().parameters(), lr=lr, betas=(beta1, beta2))

    def get_info(self):

        infos = super(AdamOptimizer, self).get_info()

        infos += f'Max learning rate: {self.max_lr}\n'
        infos += f'Betas: {(self.beta1, self.beta2)}'

        return infos
