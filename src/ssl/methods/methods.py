 ################################################################################
#   Libraries                                                                  #
################################################################################

import os
import pickle
import numpy as np
from vars import *
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.metrics import classification_report

import utils

################################################################################
#   Parent train class                                                         #
################################################################################


class SSLMethodClass:

    def set_hyperparameters(self, hyperparameters):
        # To overload
        pass

    def init_vars(self):
        # To overload
        pass

    def update_vars(self, epoch):
        # To overload
        pass

    def set_criterion(self):
        # To overload
        pass

    def cuda(self):
        # To overload
        pass

    def get_info(self):
        # To be overloaded
        pass

    def __init__(self, hyperparameters, nb_img_train, nb_classes, percent_labeled, nb_batches, batch_size, verbose, cuda):

        self.batch_idx = 0

        self.nb_img_train = nb_img_train
        self.nb_classes = nb_classes
        self.percent_labeled = percent_labeled
        self.nb_batches = nb_batches
        self.batch_size = batch_size
        self.verbose = verbose
        self.cuda_var = cuda

        self.unsup_weight_scheduler = utils.WeightSchedule(hyperparameters['unsup_loss_ramp_up_epochs'], hyperparameters['unsup_loss_ramp_up_epochs'])

        self.set_hyperparameters(hyperparameters)
        self.init_vars()
        self.set_criterion()

        if self.cuda_var:
            self.cuda()

    def epoch(self, train_dataloader, model, optimizer, epoch_id, epochs, start_epoch_id):

        model.train()

        loss_epoch = torch.tensor([0.], requires_grad=False)
        sup_loss_epoch = torch.tensor([0.], requires_grad=False)
        unsup_loss_epoch = torch.tensor([0.], requires_grad=False)
        outputs = torch.zeros(self.nb_img_train, self.nb_classes).float()

        if self.cuda_var:
            loss_epoch = loss_epoch.cuda()
            sup_loss_epoch = sup_loss_epoch.cuda()
            unsup_loss_epoch = unsup_loss_epoch.cuda()
            outputs = outputs.cuda()

        if self.verbose:
            pbar = tqdm(enumerate(train_dataloader))
        else:
            pbar = enumerate(train_dataloader)

        for batch_idx, (data, target) in pbar:

            self.batch_idx = batch_idx

            if self.cuda_var:
                data, target = data.cuda(), target.cuda()

            output = model.forward(data)

            optimizer.zero_grad()
            loss, sup_loss, unsup_loss = self.get_loss(output, target)
            loss.backward()
            optimizer.step()

            outputs[batch_idx * self.batch_size: (batch_idx + 1) * self.batch_size] = output.data.clone()
            loss_epoch += loss.detach()
            sup_loss_epoch += sup_loss.detach()
            unsup_loss_epoch += unsup_loss.detach()

            if batch_idx % LOG_INTERVAL == 0 and self.verbose:
                pbar.set_description('Train Epoch: {}/{} [{}/{} ({:.0f}%)]. Loss: {:.8f}'.format(epoch_id,
                                                                                                 epochs + start_epoch_id,
                                                                                                 batch_idx * len(data),
                                                                                                 self.nb_img_train,
                                                                                                 100. * batch_idx / self.nb_batches,
                                                                                                 (loss_epoch / (batch_idx + 1)).item()))

            if batch_idx + 1 >= self.nb_batches and self.verbose:
                pbar.set_description('Train Epoch: {}/{} [{}/{} ({:.0f}%)]. Loss: {:.8f}'.format(epoch_id,
                                                                                                 epochs + start_epoch_id,
                                                                                                 self.nb_img_train,
                                                                                                 self.nb_img_train,
                                                                                                 100.,
                                                                                                 (loss_epoch / self.nb_batches).item()))
        return outputs, loss_epoch, sup_loss_epoch, unsup_loss_epoch

    def eval(self, valuation_dataloader, model):

        model.eval()

        real_labels = []
        pred_labels = []

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(valuation_dataloader):

                if self.cuda_var:
                    data, target = data.cuda(), target.cuda()

                # Make the prediciton using the already trained model
                bs, c, h, w = data.size()
                result = model.forward(data.view(-1, c, h, w))
                result = F.softmax(result, dim=1)
                pred = result.data.max(1, keepdim=True)[1]

                # Grab the predictions and the labels into arrays
                real_labels.extend(target.data.cpu().numpy())
                pred_labels.extend(pred.squeeze().cpu().numpy())

        return accuracy_score(real_labels, pred_labels)

    def train(self, train_dataloader, valuation_dataloader, model, optimizer_wrapper, nb_img_train, nb_classes, nb_batches, batch_size, epochs, trained_model_path, start_epoch_id, verbose):

        self.start_epoch_id = start_epoch_id
        self.epochs = epochs

        logs_path = os.path.join(trained_model_path, 'logs')
        if not os.path.exists(logs_path):
            os.makedirs(logs_path)

        graphs_path = os.path.join(trained_model_path, 'graphs')
        if not os.path.exists(graphs_path):
            os.makedirs(graphs_path)

        if start_epoch_id == 0:
            torch.save({'epoch': 0,
                        'state_dict': model.state_dict()},
                       os.path.join(logs_path, f'checkpoint_0.pth'))
            accuracy = []
            losses = []
            sup_losses = []
            unsup_losses = []
            utils.save_graphs(graphs_path, accuracy, losses, sup_losses, unsup_losses)
        else:
            with open(os.path.join(graphs_path, 'accuracy.pkl'), 'rb') as f:
                accuracy = pickle.load(f)
            with open(os.path.join(graphs_path, 'loss.pkl'), 'rb') as f:
                losses = pickle.load(f)
            with open(os.path.join(graphs_path, 'sup_loss.pkl'), 'rb') as f:
                sup_losses = pickle.load(f)
            with open(os.path.join(graphs_path, 'unsup_loss.pkl'), 'rb') as f:
                unsup_losses = pickle.load(f)

        for epoch_id in range(1 + self.start_epoch_id, epochs + 1 + self.start_epoch_id):
            optimizer = optimizer_wrapper.get(model, self.start_epoch_id, epochs + self.start_epoch_id)
            self.epoch_output, loss, sup_loss, unsup_loss = self.epoch(train_dataloader, model, optimizer, epoch_id, epochs, self.start_epoch_id)
            self.update_vars(epoch_id)

            accuracy.append(self.eval(valuation_dataloader, model))
            losses.append(loss / (nb_img_train / batch_size))
            sup_losses.append(sup_loss / (nb_img_train / batch_size))
            unsup_losses.append(unsup_loss / (nb_img_train / batch_size))

            if epoch_id % TRAIN_STEP == 0:
                utils.update_checkpoint(model, epoch_id, logs_path)
                utils.save_graphs(graphs_path, accuracy, losses, sup_losses, unsup_losses)

        if epoch_id % TRAIN_STEP != 0:
            utils.update_checkpoint(model, epoch_id, logs_path)
            utils.save_graphs(graphs_path, accuracy, losses, sup_losses, unsup_losses)

################################################################################
#   Children train classes                                                     #
################################################################################


class TemporalEnsemblingClass(SSLMethodClass):

    def set_hyperparameters(self, hyperparameters):
        self.alpha = hyperparameters['alpha']
        self.ramp_epochs = hyperparameters['unsup_loss_ramp_up_epochs']
        self.ramp_mult = hyperparameters['unsup_loss_ramp_up_mult']
        self.unsup_loss_max_weight = hyperparameters['unsup_loss_max_weight'] * self.percent_labeled

    def init_vars(self):
        self.y_ema = torch.zeros(self.nb_img_train, self.nb_classes).float()
        self.unsup_weight = torch.autograd.Variable(torch.FloatTensor([0.]), requires_grad=False)

    def update_vars(self, epoch_id):
        # Updating y_ema
        self.y_ema = (self.alpha * self.y_ema + (1 - self.alpha) * F.softmax(self.epoch_output, dim=1)) / (1 - self.alpha ** epoch_id)
        # Updating unsup weight
        self.unsup_weight = self.unsup_loss_max_weight * self.unsup_weight_scheduler.step(self.epochs, self.start_epoch_id)

    def set_criterion(self):
        self.loss_sup = torch.nn.CrossEntropyLoss(reduction='sum', ignore_index=DATA_NO_LABEL)
        self.loss_unsup = torch.nn.MSELoss(reduction='mean')

    def get_loss(self, output, labels):
        y_ema_batch = Variable(self.y_ema[self.batch_idx * self.batch_size: (self.batch_idx + 1) * self.batch_size], requires_grad=False)
        loss_sup = self.loss_sup(output, labels) / self.batch_size
        loss_unsup = self.unsup_weight * self.loss_unsup(F.softmax(output), y_ema_batch)
        return loss_sup + loss_unsup, loss_sup, loss_unsup

    def cuda(self):
        self.y_ema = self.y_ema.cuda()
        self.unsup_weight = self.unsup_weight.cuda()
        self.loss_sup = self.loss_sup.cuda()
        self.loss_unsup = self.loss_unsup.cuda()

    def get_info(self):
        info_string = 'Method: Temporal Ensembling\n'
        info_string += f'Alpha: {self.alpha}\n'
        info_string += 'Unsupervised loss max weight (corrected with percent labels): {:.1f}\n'.format(self.unsup_loss_max_weight)
        info_string += f'Ramp epochs: {self.ramp_epochs}\n'
        info_string += f'Ramp mult: {self.ramp_mult}\n'
        return info_string

    def __init__(self, hyperparameters, nb_img_train, nb_classes, percent_labeled, nb_batches, batch_size, verbose, cuda):
        SSLMethodClass.__init__(self, hyperparameters, nb_img_train, nb_classes, percent_labeled, nb_batches, batch_size, verbose, cuda)

################################################################################
#   Testing class                                                              #
################################################################################


class TestingClass:

    def __init__(self, verbose, cuda):

        self.verbose = verbose
        self.cuda_var = cuda

    def test_run(self, test_dataloader, model):

        model.eval()

        real_labels = []
        pred_labels = []

        pbar = enumerate(test_dataloader)

        with torch.no_grad():
            for batch_idx, (data, target) in pbar:

                if self.cuda_var:
                    data, target = data.cuda(), target.cuda()

                # Make the prediciton using the already trained model
                bs, c, h, w = data.size()
                result = model.forward(data.view(-1, c, h, w))
                pred = result.data.max(1, keepdim=True)[1]

                # Grab the predictions and the labels into arrays
                real_labels.extend(target.data.cpu().numpy())
                pred_labels.extend(pred.squeeze().cpu().numpy())

        return pred_labels, real_labels

    def test(self, test_dataloader, model, nb_runs):

        list_classification_reports = []
        dict_metrics_scores = {}

        if self.verbose:
            pbar = tqdm(range(nb_runs))
        else:
            pbar = range(nb_runs)

        for i in pbar:
            pred_labels, real_labels = self.test_run(test_dataloader, model)
            list_classification_reports.append(classification_report(real_labels, pred_labels, digits=3, output_dict=True))
            for metric_funct in METRICS.keys():
                if metric_funct not in dict_metrics_scores.keys():
                    dict_metrics_scores[metric_funct] = 0.
                dict_metrics_scores[metric_funct] += METRICS[metric_funct](pred_labels, real_labels)

        full_classification_report = utils.avg_classifications_reports(list_classification_reports)
        for metric_funct in dict_metrics_scores.keys():
            dict_metrics_scores[metric_funct] /= (i + 1)

        return full_classification_report, dict_metrics_scores
