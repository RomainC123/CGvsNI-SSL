import numpy as np

class WeightSchedule:

    def __init__(self, ramp_up_epochs, ramp_up_mult, ramp_down_epochs=0, ramp_down_mult=0, start_epoch=0):

        self.ramp_up_epochs = ramp_up_epochs
        self.ramp_up_mult = ramp_up_mult
        self.ramp_down_epochs = ramp_down_epochs
        self.ramp_down_mult = ramp_down_mult

        self.weight = 0.
        self.epoch = start_epoch
        self.ramp_down = 0

    def step(self, total_epochs, start_epoch=0):

        self.epoch += 1

        if self.epoch <= self.ramp_up_epochs:
            self.weight = np.exp(-self.ramp_up_mult * (1 - self.epoch / self.ramp_up_epochs) ** 2)
        elif self.epoch >= total_epochs - self.ramp_down_epochs and self.ramp_down_epochs != 0:
            self.weight = np.exp(-self.ramp_down_mult * (self.ramp_down / self.ramp_down_epochs) ** 2)
            self.ramp_down += 1
        else:
            self.weight = 1.

        return self.weight
