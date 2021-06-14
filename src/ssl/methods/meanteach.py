import torch
import torch.nn.functional as F

from .base import BaseMethod
from ..utils.models import MODELS
from ..utils.constants import DATA_NO_LABEL
from ..utils.schedules import UNSUP_WEIGHT_SCHEDULE


class MeanTeacher(BaseMethod):

    def __init__(self, **kwargs):

        self.name = 'Mean Teacher'
        self._init_teacher(kwargs['model'])

        self.sup_loss = torch.nn.CrossEntropyLoss(reduction='sum', ignore_index=DATA_NO_LABEL)
        self.unsup_loss = torch.nn.MSELoss(reduction='mean')

        super(MeanTeacher, self).__init__(**kwargs)

    def cuda(self):
        self.cuda_state = True
        self.teacher_model.cuda()
        self.sup_loss = self.sup_loss.cuda()
        self.unsup_loss = self.unsup_loss.cuda()

    def _init_teacher(self, model):
        params = model.get_params()
        self.teacher_model = MODELS[params[0]](params[1], params[2])
        for param in self.teacher_model.parameters():
            param.detach_()

    def _set_hyperparameters(self, **kwargs):
        self.ema_teacher = kwargs['ema_teacher']
        self.max_unsup_weight = kwargs['max_unsup_weight'] * self.percent_labeled
        self.unsup_weight_schedule = UNSUP_WEIGHT_SCHEDULE

    def _get_hyperparameters_info(self):
        infos = f'Ema teacher: {self.ema_teacher}\n'
        infos += 'Unsupervised loss max weight: {:.1f}\n'.format(self.max_unsup_weight)
        infos += self.unsup_weight_schedule.get_info()

        return infos

    def _init_vars(self):
        self.unsup_weight = torch.autograd.Variable(torch.FloatTensor([0.]), requires_grad=False)
        if self.cuda_state:
            self.unsup_weight = self.unsup_weight.cuda()

    def _update_vars(self, epoch, total_epochs, model, output):
        self._update_teacher(model, epoch)
        self.unsup_weight = self.max_unsup_weight * UNSUP_WEIGHT_SCHEDULE(epoch, total_epochs)

    def _update_teacher(self, model, epoch):
        alpha = min(1 - 1 / epoch, self.ema_teacher)
        for teacher_param, param in zip(self.teacher_model.parameters(), model.parameters()):
            teacher_param.data.mul_(alpha).add_(1 - alpha, param.data)

    def _get_loss(self, input, output, target, idxes, batch_idx):
        self.teacher_model.train()
        with torch.no_grad():
            output_teacher = self.teacher_model.forward(input)
            output_teacher.detach_()

        sup_loss = self.sup_loss(output, target) / self.batch_size
        unsup_loss = self.unsup_weight * self.unsup_loss(F.softmax(output, dim=1), F.softmax(output_teacher, dim=1))

        return sup_loss + unsup_loss, sup_loss, unsup_loss
