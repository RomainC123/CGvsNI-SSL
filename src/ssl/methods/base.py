import os
import pickle
import numpy as np
import torch
from tqdm import tqdm

import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.metrics import classification_report

from ..utils.funtionalities import METRICS
from ..utils.constants import TRAIN_STEP, LOG_INTERVAL
from ..utils.tools import get_latest_log


class BaseMethod:

    def __init__(self, hyperparameters):

        self.cuda_state = False

        self._set_hyperparameters(hyperparameters)

    def cuda(self):
        self.cuda_state = True

    def _set_hyperparameters(self, hyperparameters):
        # TO OVERLOAD
        pass

    def _get_hyperparams_info(self):
        # TO OVERLOAD
        pass

    def _init_paths(self, trained_model_path):
        self._logs_path = os.path.join(trained_model_path, 'logs')
        if not os.path.exists(logs_path):
            os.makedirs(logs_path)
        self._graphs_path = os.path.join(trained_model_path, 'graphs')
        if not os.path.exists(graphs_path):
            os.makedirs(graphs_path)

    def _init_graphs(self, start_epoch):
        if start_epoch == 0:
            self.metrics = {}
            self.metrics.keys() = METRICS.keys()
            print(self.metrics)
            self.losses, self.sup_losses, self.unsup_losses = [], [], []
            self._save_graphs()
        else:
            self.metrics, self.losses, self.sup_losses, self.unsup_losses = self._load_graphs()

    def _update_graphs(self, metrics, losses, sup_losses, unsup_losses):
        for key in self.metrics.keys():
            self.metrics[key].append(metrics[key])
        self.losses.append(losses)
        self.sup_losses.append(sup_losses)
        self.unsup_losses.append(unsup_losses)

    def _load_graphs(self):
        with open(os.path.join(self._graphs_path, 'metrics.pkl'), 'rb') as f:
            metrics = pickle.load(f)
        with open(os.path.join(self._graphs_path, 'loss.pkl'), 'rb') as f:
            losses = pickle.load(f)
        with open(os.path.join(self._graphs_path, 'sup_loss.pkl'), 'rb') as f:
            sup_losses = pickle.load(f)
        with open(os.path.join(self._graphs_path, 'unsup_loss.pkl'), 'rb') as f:
            unsup_losses = pickle.load(f)

        return metrics, losses, sup_losses, unsup_losses

    def _save_graphs(self):
        with open(os.path.join(self._graphs_path, 'metrics.pkl'), 'wb') as f:
            pickle.dump(self.metrics, f)
        with open(os.path.join(self._graphs_path, 'loss.pkl'), 'wb') as f:
            pickle.dump(self.losses, f)
        with open(os.path.join(self._graphs_path, 'sup_loss.pkl'), 'wb') as f:
            pickle.dump(self.sup_losses, f)
        with open(os.path.join(self._graphs_path, 'unsup_loss.pkl'), 'wb') as f:
            pickle.dump(self.unsup_losses, f)

    def _init_vars(self):
        # TO OVERLOAD
        pass

    def _update_vars(self, epoch):
        # TO OVERLOAD
        pass

    def _update_checkpoint(self, model, epoch):
        latest_log, latest_log_epoch = get_latest_log(self._logs_path)
        torch.save({'epoch': epoch,
                    'state_dict': model.state_dict()},
                   os.path.join(self._logs_path, f'checkpoint_{epoch}.pth'))
        os.remove(os.path.join(self._logs_path, latest_log))

    def _init_vars_epoch(self, nb_samples_train, nb_classes):
        loss_epoch = torch.tensor([0.], requires_grad=False)
        sup_loss_epoch = torch.tensor([0.], requires_grad=False)
        unsup_loss_epoch = torch.tensor([0.], requires_grad=False)
        outputs = torch.zeros(nb_samples_train, nb_classes).float()

        return loss_epoch, sup_loss_epoch, unsup_loss_epoch, outputs

    def _get_loss(self, output, target):
        # TO OVERLOAD
        pass

    def _save_train_info(self, batch_size, nb_batches, start_epoch, total_epochs):

        train_info = f'Batch size: {batch_size}\n'
        train_info += f'Number of batches: {nb_batches}\n'
        train_info += f'Starting epoch: {start_epoch}\n'
        train_info += f'Total number of epochs: {total_epochs}\n'

        with open('info.txt', 'a+') as f:
            f.write(train_info)
        if self.verbose:
            print(train_info)

    def train(self, train_dataloader, valuation_dataloader, model, optimizer, start_epoch, total_epochs, trained_model_path, verbose):
        # Grab from objects: nb_img_train, nb_classes, nb_batches, batch_size

        self.verbose = verbose
        batch_size = train_dataloader.batch_size
        nb_batches = len(train_dataloader)

        self._init_paths(trained_model_path)
        self._init_graphs(start_epoch)
        self._init_vars()

        self._save_train_info(batch_size, nb_batches, start_epoch, total_epochs)

        for epoch in range(1 + start_epoch, 1 + total_epochs):

            optimizer_epoch = optimizer(model, epoch, total_epochs)
            output, losses, sup_losses, unsup_losses = self.epoch(train_dataloader, model, optimizer, epoch, total_epochs)

            self._update_graphs(metrics, losses, sup_losses, unsup_losses)
            self._update_vars(output, epoch)

            if epoch_id % TRAIN_STEP == 0:
                self._update_checkpoint(model, epoch)
                self._save_graphs(metrics, losses, sup_losses, unsup_losses)

        if epoch_id % TRAIN_STEP != 0:
            self._update_checkpoint(model, epoch)
            self._save_graphs(metrics, losses, sup_losses, unsup_losses)

    def epoch(self, train_dataloader, model, optimizer, epoch, total_epochs):

        model.train()

        nb_samples_train = train_dataloader.nb_samples_train
        nb_classes = train_dataloader.nb_classes
        batch_size = train_dataloader.batch_size
        nb_batches = len(train_dataloader)

        loss_epoch, sup_loss_epoch, unsup_loss_epoch, outputs = self._init_vars_epoch(nb_samples_train, nb_classes)

        if self.cuda_state:
            loss_epoch = loss_epoch.cuda()
            sup_loss_epoch = sup_loss_epoch.cuda()
            unsup_loss_epoch = unsup_loss_epoch.cuda()
            outputs = outputs.cuda()

        if self.verbose:
            pbar = tqdm(enumerate(train_dataloader))
        else:
            pbar = enumerate(train_dataloader)

        for batch_idx, (data, target) in pbar:

            if self.cuda_state:
                data, target = data.cuda(), target.cuda()

            output = model().forward(data)

            optimizer.zero_grad()
            loss, sup_loss, unsup_loss = self._get_loss(output, target)
            loss.backward()
            optimizer.step()

            loss_epoch += loss.detach()
            sup_loss_epoch += sup_loss.detach()
            unsup_loss_epoch += unsup_loss.detach()
            outputs[batch_idx * batch_size: (batch_idx + 1) * batch_size] = output.data.clone()

            if batch_idx % LOG_INTERVAL == 0 and self.verbose:
                pbar.set_description('Train Epoch: {}/{} [{}/{} ({:.0f}%)]. Loss: {:.8f}'.format(epoch,
                                                                                                 total_epochs,
                                                                                                 batch_idx * len(data),
                                                                                                 nb_img_train,
                                                                                                 100. * batch_idx / nb_batches,
                                                                                                 (loss_epoch / (batch_idx + 1)).item()))

        if batch_idx + 1 >= nb_batches and self.verbose:
            pbar.set_description('Train Epoch: {}/{} [{}/{} ({:.0f}%)]. Loss: {:.8f}'.format(epoch,
                                                                                             total_epochs,
                                                                                             nb_img_train,
                                                                                             nb_img_train,
                                                                                             100.,
                                                                                             (loss_epoch / nb_batches).item()))
        return outputs, loss_epoch, sup_loss_epoch, unsup_loss_epoch

    def eval(self, valuation_dataloader, model):

        model.eval()

        real_labels = []
        pred_labels = []

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(valuation_dataloader):

                if self.cuda_state:
                    data, target = data.cuda(), target.cuda()

                # Make the prediciton using the already trained model
                batch_size, c, h, w = data.size()
                output = model.forward(data.view(-1, c, h, w))
                result = F.softmax(output, dim=1)
                pred = result.data.max(1, keepdim=True)[1]

                # Grab the predictions and the labels into arrays
                real_labels.extend(target.data.cpu().numpy())
                pred_labels.extend(pred.squeeze().cpu().numpy())

        return accuracy_score(real_labels, pred_labels)

    def get_method_info(self):

        infos = f'Method: {self.name}\n'
        infos += self._get_hyperparams_info()

        return infos
