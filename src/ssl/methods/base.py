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

class SSLMethodClass:

    @abstractmethod
    def _set_hyperparameters(self, hyperparameters):
        pass

    @abstractmethod
    def _init_vars(self):
        pass

    @abstractmethod
    def _update_vars(self, epoch):
        pass

    @abstractmethod
    def cuda(self):
        # To overload
        pass

    @abstractmethod
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

        self.set_hyperparameters(hyperparameters)
        self.init_vars()

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
