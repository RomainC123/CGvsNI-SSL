import torch
import torch.nn.functional as F
from .base import BaseMethod
from ..utils.schedules import UNSUP_WEIGHT_SCHEDULE
from ..utils.constants import DATA_NO_LABEL


class TemporalEnsembling(BaseMethod):

    def __init__(self, **kwargs):

        self.name = 'Temporal Ensembling'

        super(TemporalEnsembling, self).__init__(**kwargs)

    def _set_hyperparameters(self, **kwargs):
        self.alpha = kwargs['alpha']
        self.max_unsup_weight = kwargs['max_unsup_weight']
        self.unsup_weight_schedule = UNSUP_WEIGHT_SCHEDULE
        self.ramp_epochs = self.unsup_weight_schedule.ramp_up_epochs
        self.ramp_mult = self.unsup_weight_schedule.ramp_up_mult

    def _get_hyperparameters_info(self):
        infos = f'Alpha: {self.alpha}\n'
        infos += 'Unsupervised loss max weight (uncorrected): {:.1f}\n'.format(self.max_unsup_weight)
        infos += self.unsup_weight_schedule.get_info()

        return infos

    def _init_vars(self):
        self.y_ema = torch.zeros(self.nb_samples_train, self.nb_classes).float()
        self.unsup_weight = torch.autograd.Variable(torch.FloatTensor([0.]), requires_grad=False)
        if self.cuda_state:
            self.y_ema = self.y_ema.cuda()
            self.unsup_weight = self.unsup_weight.cuda()

    def _update_vars(self, output, epoch, total_epochs):
        self.y_ema = (self.alpha * self.y_ema + (1 - self.alpha) * output) / (1 - self.alpha ** epoch)
        self.unsup_weight = self.max_unsup_weight * UNSUP_WEIGHT_SCHEDULE(epoch, total_epochs)

    def _get_loss(self, output, target, idxes, batch_idx):

        def masked_crossentropy(out, labels):
            nbsup = len(torch.nonzero(labels >= 0))
            loss = F.cross_entropy(out, labels, size_average=False, ignore_index=DATA_NO_LABEL)
            if nbsup != 0:
                loss = loss / nbsup
            return loss, nbsup

        def mse_loss(out1, out2):
            quad_diff = torch.sum((F.softmax(out1, dim=1) - F.softmax(out2, dim=1)) ** 2)
            return quad_diff / out1.data.nelement()

        y_ema_batch = torch.autograd.Variable(self.y_ema[idxes], requires_grad=False)
        sup_loss, nbsup = masked_crossentropy(output, target)
        unsup_loss = self.unsup_weight * mse_loss(output, y_ema_batch)

        return sup_loss + unsup_loss, sup_loss, unsup_loss
