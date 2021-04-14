import torch
from .base import BaseMethod
from ..utils.schedules import UNSUP_WEIGHT_SCHEDULE
from ..utils.constants import DATA_NO_LABEL


class TemporalEnsembling(BaseMethod):

    def __init__(self, hyperparameters):

        self.name = 'Temporal Ensembling'

        super(TemporalEnsembling, self).__init__(hyperparameters)

    def _set_hyperparameters(self, hyperparameters):
        self.alpha = hyperparameters['alpha']
        self.ramp_epochs = hyperparameters['unsup_loss_ramp_up_epochs']
        self.ramp_mult = hyperparameters['unsup_loss_ramp_up_mult']
        self.unsup_loss_max_weight = hyperparameters['unsup_loss_max_weight']
        self.unsup_weight_schedule = UNSUP_WEIGHT_SCHEDULE

    def _get_hyperparameters_info(self):
        infos = f'Alpha: {self.alpha}\n'
        infos += 'Unsupervised loss max weight (uncorrected): {:.1f}\n'.format(self.unsup_loss_max_weight)
        infos += f'Ramp epochs: {self.ramp_epochs}\n'
        infos += f'Ramp mult: {self.ramp_mult}\n'

        return infos

    def _init_vars(self):
        self.y_ema = torch.zeros(self.nb_samples_train, self.nb_classes).float()
        self.unsup_weight = torch.autograd.Variable(torch.FloatTensor([0.]), requires_grad=False)
        if self.cuda_state:
            self.y_ema = self.y_ema.cuda()
            self.unsup_weight = self.unsup_weight.cuda()

    def _update_vars(self, output, epoch, total_epochs):
        self.y_ema = (self.alpha * self.y_ema + (1 - self.alpha) * output) / (1 - self.alpha ** epoch)
        self.unsup_weight = self.unsup_loss_max_weight * UNSUP_WEIGHT_SCHEDULE(epoch, total_epochs)

    def _get_loss(self, output, target, batch_idx):
        sup_loss_f = torch.nn.CrossEntropyLoss(reduction='sum', ignore_index=DATA_NO_LABEL)
        unsup_loss_f = torch.nn.MSELoss(reduction='mean')
        y_ema_batch = torch.autograd.Variable(self.y_ema[batch_idx * self.batch_size: (batch_idx + 1) * self.batch_size], requires_grad=False)
        if self.cuda_state:
            sup_loss_f = sup_loss_f.cuda()
            unsup_loss_f = unsup_loss_f.cuda()
        sup_loss = sup_loss_f(output, target) / self.batch_size
        unsup_loss = self.unsup_weight * self.percent_labeled * unsup_loss_f(output, y_ema_batch)

        return sup_loss + unsup_loss, sup_loss, unsup_loss
