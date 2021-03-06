import torch
import torch.nn.functional as F

from .base import BaseMethod
from ..utils.constants import DATA_NO_LABEL
from ..utils.schedules import UNSUP_WEIGHT_SCHEDULE


class TemporalEnsembling(BaseMethod):

    def __init__(self, **kwargs):

        self.name = 'Temporal Ensembling'

        self.sup_loss = torch.nn.CrossEntropyLoss(reduction='sum', ignore_index=DATA_NO_LABEL)
        self.unsup_loss = torch.nn.MSELoss(reduction='mean')

        super(TemporalEnsembling, self).__init__(**kwargs)

    def cuda(self):
        self.cuda_state = True
        self.sup_loss = self.sup_loss.cuda()
        self.unsup_loss = self.unsup_loss.cuda()

    def _set_hyperparameters(self, **kwargs):
        self.alpha = kwargs['alpha']
        self.max_unsup_weight = kwargs['max_unsup_weight'] * self.percent_labeled
        self.unsup_weight_schedule = UNSUP_WEIGHT_SCHEDULE

    def _get_hyperparameters_info(self):
        infos = f'Alpha: {self.alpha}\n'
        infos += 'Unsupervised loss max weight: {:.1f}\n'.format(self.max_unsup_weight)
        infos += self.unsup_weight_schedule.get_info()

        return infos

    def _init_vars(self):
        self.y_ema = torch.zeros(self.nb_samples_train, self.nb_classes).float()
        self.ensemble_prediction = torch.zeros(self.nb_samples_train, self.nb_classes).float()
        self.unsup_weight = torch.autograd.Variable(torch.FloatTensor([0.]), requires_grad=False)
        if self.cuda_state:
            self.y_ema = self.y_ema.cuda()
            self.ensemble_prediction = self.ensemble_prediction.cuda()
            self.unsup_weight = self.unsup_weight.cuda()

    def _update_vars(self, epoch, total_epochs, model, output):
        self.ensemble_prediction.data = self.alpha * self.ensemble_prediction.data + (1 - self.alpha) * F.softmax(output.data, dim=1)
        self.y_ema.data = self.ensemble_prediction.data / (1 - self.alpha ** epoch)
        self.unsup_weight = self.max_unsup_weight * UNSUP_WEIGHT_SCHEDULE(epoch, total_epochs)

    def _get_loss(self, model, data, target, idxes, batch_idx):

        output = model.forward(data)
        y_ema_batch = torch.autograd.Variable(self.y_ema[idxes], requires_grad=False)
        sup_loss = self.sup_loss(output, target) / self.batch_size
        unsup_loss = self.unsup_weight * self.unsup_loss(F.softmax(output, dim=1), y_ema_batch)

        return sup_loss + unsup_loss, sup_loss, unsup_loss, output
