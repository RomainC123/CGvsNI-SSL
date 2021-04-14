################################################################################
#   Base Optimizer class                                                       #
################################################################################

class BaseOptimizerContainer:

    def __init__(self, hyperparameters):
        # TO COMPLETE
        pass

    def __call__(self, model, epoch, total_epochs):
        # TO OVERLOAD
        pass

    def get_info(self):

        infos = f'Optimizer: {self.name}\n'

        return infos
