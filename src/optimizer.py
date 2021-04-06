from vars import *
from utils import WeightSchedule

from torch.optim import Adam

IMPLEMENTED_OPTIMIZERS = {
    'Adam': Adam
}


class OptimizerWrapper:

    def __init__(self, name, max_lr=0.001):

        self.name = name

        self.ramp_up_epochs = OPTIMIZER_PARAMS[self.name]['ramp_up_epochs']
        self.ramp_up_mult = OPTIMIZER_PARAMS[self.name]['ramp_up_mult']
        self.ramp_down_epochs = OPTIMIZER_PARAMS[self.name]['ramp_down_epochs']
        self.ramp_down_mult = OPTIMIZER_PARAMS[self.name]['ramp_down_mult']
        self.lr_schedule = WeightSchedule(self.ramp_up_epochs, self.ramp_up_mult, self.ramp_down_epochs, self.ramp_down_mult)
        self.max_lr = OPTIMIZER_PARAMS[self.name]['max_lr']

        if name in IMPLEMENTED_OPTIMIZERS.keys():
            self.optimizer_function = IMPLEMENTED_OPTIMIZERS[name]
        else:
            raise RuntimeError('Optimizer not implemented: ', name)

    def get_info(self):

        infos = f'\nOptimizer: {self.name}\n'
        infos += f'Max learning rate: {self.max_lr}\n'
        infos += f'Ramp up epochs: {self.ramp_up_epochs}\n'
        infos += f'Ramp up mult: {self.ramp_up_mult}\n'
        infos += f'Ramp down epochs: {self.ramp_down_epochs}\n'
        infos += f'Ramp down mult: {self.ramp_down_mult}\n'

        return infos

    def get(self, model, start_epoch, total_epochs):

        self.lr = self.lr_schedule.step(start_epoch, total_epochs) * self.max_lr
        return self.optimizer_function(model.parameters(), lr=self.lr, betas=(0.9, 0.999))
