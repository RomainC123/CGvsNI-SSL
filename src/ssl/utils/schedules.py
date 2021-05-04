import numpy as np
from .hyperparameters import SCHEDULES_DEFAULT


class WeightSchedule:

    def __init__(self, **kwargs):

        self.ramp_up_epochs = kwargs['ramp_up_epochs']
        self.ramp_up_mult = kwargs['ramp_up_mult']
        self.ramp_down_epochs = kwargs['ramp_down_epochs']
        self.ramp_down_mult = kwargs['ramp_down_mult']

        self.lower = kwargs['lower']
        self.upper = kwargs['upper']

    def __call__(self, epoch, total_epochs):

        if epoch <= self.ramp_up_epochs:
            weight = np.exp(-self.ramp_up_mult * (1 - epoch / self.ramp_up_epochs) ** 2)
        elif epoch >= total_epochs - self.ramp_down_epochs and self.ramp_down_epochs != 0:
            ramp_down = epoch - (total_epochs - self.ramp_down_epochs)
            weight = np.exp(-self.ramp_down_mult * (ramp_down / self.ramp_down_epochs) ** 2)
        else:
            weight = 1.

        return self.lower + weight * (self.upper - self.lower)

    def get_info(self):

        infos = ''

        if self.ramp_up_epochs != 0:
            infos += f'Ramp up epochs: {self.ramp_up_epochs}\n'
            infos += f'Ramp up mult: {self.ramp_up_mult}\n'
        if self.ramp_down_epochs != 0:
            infos += f'Ramp down epochs: {self.ramp_down_epochs}\n'
            infos += f'Ramp down mult: {self.ramp_down_mult}\n'
        infos += f'Lower bound: {self.lower}\n'
        infos += f'Upper bound: {self.upper}'

        return infos


LR_SCHEDULE = WeightSchedule(**SCHEDULES_DEFAULT['lr'])
B1_SCHEDULE = WeightSchedule(**SCHEDULES_DEFAULT['beta1'])
UNSUP_WEIGHT_SCHEDULE = WeightSchedule(**SCHEDULES_DEFAULT['unsup_weight'])
