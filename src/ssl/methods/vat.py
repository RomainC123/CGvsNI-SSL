import torch
import torch.nn.functional as F

from .base import BaseMethod
from ..utils.constants import DATA_NO_LABEL
from ..utils.schedules import UNSUP_WEIGHT_SCHEDULE
from ..utils.vat import VATLoss


class VAT(BaseMethod):

    def __init__(self, **kwargs):

        self.name = 'Virtual Adversarial Training'

        self.sup_loss = torch.nn.CrossEntropyLoss(reduction='sum', ignore_index=DATA_NO_LABEL)
        self.vat_loss = VATLoss(xi=10.0, eps=1.0, ip=1)

        super(VAT, self).__init__(**kwargs)

    def cuda(self):
        self.cuda_state = True
        self.sup_loss = self.sup_loss.cuda()
        self.vat_loss = self.vat_loss.cuda()

    def _set_hyperparameters(self, **kwargs):
        self.max_unsup_weight = kwargs['max_unsup_weight'] * self.percent_labeled
        self.unsup_weight_schedule = UNSUP_WEIGHT_SCHEDULE

    def _get_hyperparameters_info(self):
        infos += 'Unsupervised loss max weight: {:.1f}\n'.format(self.max_unsup_weight)
        infos += self.unsup_weight_schedule.get_info()

        return infos

    def _init_vars(self):
        self.unsup_weight = torch.autograd.Variable(torch.FloatTensor([0.]), requires_grad=False)
        if self.cuda_state:
            self.unsup_weight = self.unsup_weight.cuda()

    def _update_vars(self, epoch, total_epochs, model, output):
        self.unsup_weight = self.max_unsup_weight * UNSUP_WEIGHT_SCHEDULE(epoch, total_epochs)

    def _get_loss(self, model, data, target, idxes, batch_idx):
        
        vat_loss = self.unsup_weight * self.vat_loss(model, data)
        output = model.forward(data)
        sup_loss = self.sup_loss(output, target) / self.batch_size

        return sup_loss + unsup_loss, sup_loss, unsup_loss
