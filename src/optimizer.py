from torch.optim import Adam

IMPLEMENTED_OPTIMIZERS = {
    'Adam': Adam
}


class Optimizer:

    def __init__(self, name, max_lr=0.001, ramp_up_epochs=80, ramp_up_mult=5, ramp_down_epochs=50, , ramp_down_mult=12.5):

        self.weight_schedule = utils.get_weight_ramp_up_down()

        if name in IMPLEMENTED_OPTIMIZERS.keys():
            self.function = IMPLEMENTED_OPTIMIZERS[name]()
        else:
            raise RuntimeError('Optimizer not implemented: ', name)
