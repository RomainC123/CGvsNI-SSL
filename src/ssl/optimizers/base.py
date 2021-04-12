################################################################################
#   Base Optimizer class                                                       #
################################################################################

class BaseOptimizer:

    def __init__(self, **kwargs):
        # TO COMPLETE
        pass

    def __call__(self, model, epoch, total_epochs):
        # TO OVERLOAD
        pass

    def get_info(self):

        infos = f'Optimizer: {self.name}\n'

        return infos
