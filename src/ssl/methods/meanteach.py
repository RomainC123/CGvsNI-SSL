import torch
import torch.nn.functional as F

from .base import BaseMethod
from ..utils.constants import DATA_NO_LABEL
from ..utils.schedules import UNSUP_WEIGHT_SCHEDULE


class MeanTeacher(BaseMethod):

    def __init__(self, **kwargs):

        self.name = 'Mean Teacher'

        self.sup_loss = torch.nn.CrossEntropyLoss(reduction='sum', ignore_index=DATA_NO_LABEL)
        self.unsup_loss = torch.nn.MSELoss(reduction='mean')

        super(TemporalEnsemblingNewLoss, self).__init__(**kwargs)

    def cuda(self):
        self.cuda_state = True
        self.sup_loss = self.sup_loss.cuda()
        self.unsup_loss = self.unsup_loss.cuda()

    def _set_hyperparameters(self, **kwargs):
        self.alpha = kwargs['alpha']
        self.max_unsup_weight = kwargs['max_unsup_weight'] * self.percent_labeled
        self.unsup_weight_schedule = UNSUP_WEIGHT_SCHEDULE

    def _get_hyperparameters_info(self):
        infos = f'Alpha teacher: {self.alpha}\n'
        infos += 'Unsupervised loss max weight: {:.1f}\n'.format(self.max_unsup_weight)
        infos += self.unsup_weight_schedule.get_info()

        return infos

    def _init_vars(self):

        self.teacher_model =
        self.unsup_weight = torch.autograd.Variable(torch.FloatTensor([0.]), requires_grad=False)
        if self.cuda_state:
            self.y_ema = self.y_ema.cuda()
            self.ensemble_prediction = self.ensemble_prediction.cuda()
            self.unsup_weight = self.unsup_weight.cuda()

    def _update_vars(self, output, epoch, total_epochs):
        slef.teacher_model._update_weights()
        self.unsup_weight = self.max_unsup_weight * UNSUP_WEIGHT_SCHEDULE(epoch, total_epochs)

    def _get_loss(self, output, target, idxes, batch_idx):

        with torch.no_grad():

        sup_loss = self.sup_loss(output, target) / self.batch_size
        unsup_loss = self.unsup_weight * self.unsup_loss(F.softmax(output, dim=1), y_ema_batch)

        return sup_loss + unsup_loss, sup_loss, unsup_loss
