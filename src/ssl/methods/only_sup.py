import torch
import torch.nn.functional as F
from .base import BaseMethod
from ..utils.constants import DATA_NO_LABEL


class OnlySup(BaseMethod):

    def __init__(self, **kwargs):

        self.name = 'Only supervised'

        self.sup_loss = torch.nn.CrossEntropyLoss(reduction='sum', ignore_index=DATA_NO_LABEL)

        super(OnlySup, self).__init__(**kwargs)

    def cuda(self):
        self.cuda_state = True
        self.sup_loss = self.sup_loss.cuda()

    def _get_hyperparameters_info(self):

        return ''

    def _get_loss(self, model, data, target, idxes, batch_idx):

        output = model.forward(data)
        sup_loss = self.sup_loss(output, target) / self.batch_size
        unsup_loss = torch.tensor(0.)

        return sup_loss + unsup_loss, sup_loss, unsup_loss, output
