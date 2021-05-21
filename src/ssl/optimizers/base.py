################################################################################
#   Base Optimizer class                                                       #
################################################################################

class BaseOptimizerContainer:

    def __init__(self, **kwargs):
        # TO COMPLETE
        pass

    def __call__(self, model, epoch, total_epochs):
        # TO OVERLOAD
        pass

    def zero_grad(self):

        self.optim.zero_grad()

    def step(self):

        self.optim.step()

    def get_info(self):

        infos = f'Optimizer: {self.name}\n'

        return infos
