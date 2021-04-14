import numpy as np

class WeightSchedule:

    def __init__(self, ramp_up_epochs, ramp_up_mult, ramp_down_epochs, ramp_down_mult):

        self.ramp_up_epochs = ramp_up_epochs
        self.ramp_up_mult = ramp_up_mult
        self.ramp_down_epochs = ramp_down_epochs
        self.ramp_down_mult = ramp_down_mult

    def __call__(self, epoch, total_epochs):

        if epoch <= self.ramp_up_epochs:
            weight = np.exp(-self.ramp_up_mult * (1 - epoch / self.ramp_up_epochs) ** 2)
        elif epoch >= total_epochs - self.ramp_down_epochs and self.ramp_down_epochs != 0:
            ramp_down = epoch - (total_epochs - self.ramp_down_epochs)
            weight = np.exp(-self.ramp_down_mult * (ramp_down / self.ramp_down_epochs) ** 2)
        else:
            weight = 1.

        return weight


LR_SCHEDULE = WeightSchedule(ramp_up_epochs=80, ramp_up_mult=5, ramp_down_epochs=50, ramp_down_mult=12.5)
B1_SCHEDULE = WeightSchedule(ramp_up_epochs=0, ramp_up_mult=0, ramp_down_epochs=0, ramp_down_mult=0)
UNSUP_WEIGHT_SCHEDULE = WeightSchedule(ramp_up_epochs=80, ramp_up_mult=5, ramp_down_epochs=0, ramp_down_mult=0)
