import os
import pickle
import numpy as np
import torch
from tqdm import tqdm

from datetime import datetime
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.metrics import classification_report

from ..utils.metrics import METRICS
from ..utils.constants import TRAIN_STEP, LOG_INTERVAL, TEST_RUNS
from ..utils.tools import get_latest_log, avg_classification_reports, get_metrics_report


class BaseMethod:

    def __init__(self, **kwargs):

        self.cuda_state = False

        self._set_hyperparameters(**kwargs)

    def cuda(self):
        self.cuda_state = True

    def _set_hyperparameters(self, **kwargs):
        # TO OVERLOAD
        pass

    def _get_hyperparams_info(self):
        # TO OVERLOAD
        pass

    def _init_paths(self, trained_model_path):
        self.main_path = trained_model_path
        if not os.path.exists(self.main_path):
            os.makedirs(self.main_path)
        self._logs_path = os.path.join(self.main_path, 'logs')
        if not os.path.exists(self._logs_path):
            os.makedirs(self._logs_path)
        self._graphs_path = os.path.join(self.main_path, 'graphs')
        if not os.path.exists(self._graphs_path):
            os.makedirs(self._graphs_path)

    def _init_checkpoint(self, model, start_epoch):
        if start_epoch == 0:
            model.save(self._logs_path, 0)
        else:
            self._load_checkpoint(model, start_epoch)

    def _init_graphs(self, start_epoch):
        if start_epoch == 0:
            self.metrics_eval = {}
            self.metrics_test = {}
            for key in METRICS.keys():
                self.metrics_eval[key] = []
                self.metrics_test[key] = []
            self.losses, self.sup_losses, self.unsup_losses = [], [], []
            self._save_graphs()
        else:
            self._load_graphs()

    def _load_checkpoint(self, model, start_epoch):
        try:
            log = open(os.path.join(self._logs_path, f'checkpoint_{start_epoch}'))
        except:
            raise RuntimeError(f'Log {start_epoch} not found')
        checkpoint = torch.load(os.path.join(self.logs_path, latest_log))
        self.model.model.load_state_dict(checkpoint['state_dict'])

    def _load_graphs(self):
        with open(os.path.join(self._graphs_path, 'metrics_eval.pkl'), 'rb') as f:
            self.metrics_eval = pickle.load(f)
        with open(os.path.join(self._graphs_path, 'metrics_test.pkl'), 'rb') as f:
            self.metrics_test = pickle.load(f)
        with open(os.path.join(self._graphs_path, 'loss.pkl'), 'rb') as f:
            self.losses = pickle.load(f)
        with open(os.path.join(self._graphs_path, 'sup_loss.pkl'), 'rb') as f:
            self.sup_losses = pickle.load(f)
        with open(os.path.join(self._graphs_path, 'unsup_loss.pkl'), 'rb') as f:
            self.unsup_losses = pickle.load(f)

    def _update_checkpoint(self, model, epoch):
        latest_log, _ = get_latest_log(self.main_path)
        torch.save({'epoch': epoch,
                    'state_dict': model.model.state_dict()},
                   os.path.join(self._logs_path, f'checkpoint_{epoch}.pth'))
        os.remove(os.path.join(self._logs_path, latest_log))

    def _update_graphs(self, metrics_eval, metrics_test, losses, sup_losses, unsup_losses):
        for key in self.metrics_eval.keys():
            self.metrics_eval[key].append(metrics_eval[key])
            self.metrics_test[key].append(metrics_test[key])
        self.losses.append(losses)
        self.sup_losses.append(sup_losses)
        self.unsup_losses.append(unsup_losses)

    def _save_graphs(self):
        with open(os.path.join(self._graphs_path, 'metrics_eval.pkl'), 'wb') as f:
            pickle.dump(self.metrics_eval, f)
        with open(os.path.join(self._graphs_path, 'metrics_test.pkl'), 'wb') as f:
            pickle.dump(self.metrics_test, f)
        with open(os.path.join(self._graphs_path, 'loss.pkl'), 'wb') as f:
            pickle.dump(self.losses, f)
        with open(os.path.join(self._graphs_path, 'sup_loss.pkl'), 'wb') as f:
            pickle.dump(self.sup_losses, f)
        with open(os.path.join(self._graphs_path, 'unsup_loss.pkl'), 'wb') as f:
            pickle.dump(self.unsup_losses, f)

    def _init_vars(self):
        # TO OVERLOAD
        pass

    def _update_vars(self, epoch, total_epochs):
        # TO OVERLOAD
        pass

    def _init_vars_epoch(self):
        loss_epoch = 0.
        sup_loss_epoch = 0.
        unsup_loss_epoch = 0.
        outputs = torch.zeros((self.nb_samples_train, self.nb_classes))

        return loss_epoch, sup_loss_epoch, unsup_loss_epoch, outputs

    def _get_loss(self, output, target, idxes, batch_idx):
        # TO OVERLOAD
        pass

    def _save_train_info(self, start_epoch, total_epochs):

        train_info = '------------------------------------\n'
        train_info += f'Batch size: {self.batch_size}\n'
        train_info += f'Number of batches: {self.nb_batches}\n'
        train_info += f'Starting epoch: {start_epoch}\n'
        train_info += f'Total number of epochs: {total_epochs}'

        with open(os.path.join(self.main_path, 'info.txt'), 'a+') as f:
            f.write('\n' + train_info)
        if self.verbose_train:
            print(train_info)

    def _save_train_duration(self, timer):

        str_time = str(timer)
        split_time = str_time.split(':')
        split_time[2] = split_time[2][:2]
        timer_info = f'Training time: {":".join(split_time)}'

        with open(os.path.join(self.main_path, 'info.txt'), 'a+') as f:
            f.write('\n' + timer_info)
        if self.verbose_train:
            print(timer_info)

    def _epoch(self, train_dataloader, preprocess, model, optimizer, epoch, total_epochs):

        print(epoch)

        model.train()

        loss_epoch, sup_loss_epoch, unsup_loss_epoch, outputs = self._init_vars_epoch()
        optimizer.update_params(epoch, total_epochs)

        if self.cuda_state:
            outputs = outputs.cuda()

        if self.verbose_train:
            pbar = tqdm(enumerate(train_dataloader))
        else:
            pbar = enumerate(train_dataloader)

        for batch_idx, (data, target, idxes) in pbar:

            if self.cuda_state:
                data, target = data.cuda(), target.cuda()

            data = preprocess(data)

            optimizer.zero_grad()
            output = model.forward(data)
            loss, sup_loss, unsup_loss = self._get_loss(output, target, idxes, batch_idx)
            loss.backward()
            optimizer.step()

            loss_epoch += loss.data.cpu().numpy()
            sup_loss_epoch += sup_loss.data.cpu().numpy()
            unsup_loss_epoch += unsup_loss.data.cpu().numpy()
            outputs[idxes] = output.data.clone()

            if batch_idx % LOG_INTERVAL == 0 and self.verbose_train:
                pbar.set_description('Train Epoch: {}/{} (lr: {:.2E}) [{}/{} ({:.0f}%)]. Loss: {:.5f} '.format(epoch,
                                                                                                               total_epochs,
                                                                                                               optimizer.optim.param_groups[0]['lr'],
                                                                                                               batch_idx * len(data),
                                                                                                               self.nb_samples_train,
                                                                                                               100. * batch_idx / self.nb_batches,
                                                                                                               (loss_epoch / (batch_idx + 1)).item()))

            if batch_idx + 1 >= self.nb_batches and self.verbose_train:
                pbar.set_description('Train Epoch: {}/{} (lr: {:.2E}) [{}/{} ({:.0f}%)]. Loss: {:.5f} '.format(epoch,
                                                                                                               total_epochs,
                                                                                                               optimizer.optim.param_groups[0]['lr'],
                                                                                                               self.nb_samples_train,
                                                                                                               self.nb_samples_train,
                                                                                                               100.,
                                                                                                               (loss_epoch / self.nb_batches).item()))
        self._update_vars(outputs, epoch, total_epochs)
        del outputs

        return loss_epoch, sup_loss_epoch, unsup_loss_epoch

    def _eval(self, dataloader, preprocess, model):

        model.eval()

        real_labels = []
        pred_labels = []

        with torch.no_grad():
            for batch_idx, (data, target, idxes) in enumerate(dataloader):

                if self.cuda_state:
                    data, target = data.cuda(), target.cuda()

                data = preprocess(data)

                output = model.forward(data)
                result = F.softmax(output, dim=1)
                pred = result.data.max(1, keepdim=True)[1]

                # Grab the predictions and the labels into arrays
                real_labels.extend(target.data.cpu().numpy())
                pred_labels.extend(pred.squeeze().cpu().numpy())

        return real_labels, pred_labels

    def _get_metrics(self, real_labels, pred_labels):
        metrics = {}
        for key in METRICS.keys():
            metrics[key] = METRICS[key](real_labels, pred_labels)

        return metrics

    def train(self, dataset, model, optimizer, start_epoch, total_epochs, trained_model_path, verbose):
        # Grab from objects: nb_img_train, nb_classes, nb_batches, batch_size

        dataloader_train, dataloader_valuation, dataloader_test = dataset.get_dataloaders(self.cuda_state)
        self.verbose_train = verbose
        self.nb_samples_train = dataset.nb_samples_train
        self.nb_classes = dataset.nb_classes
        self.percent_labeled = dataset.percent_labeled
        self.batch_size = dataloader_train.batch_size
        self.nb_batches = len(dataloader_train)

        self._init_paths(trained_model_path)
        self._init_checkpoint(model, start_epoch)
        self._init_graphs(start_epoch)
        self._init_vars()
        optimizer.create_optim(model)

        self._save_train_info(start_epoch, total_epochs)

        start_time = datetime.now()

        for epoch in range(1 + start_epoch, 1 + total_epochs):

            losses, sup_losses, unsup_losses = self._epoch(dataloader_train, dataset.preprocess, model, optimizer, epoch, total_epochs)

            losses = losses / self.nb_batches
            sup_losses = sup_losses / self.nb_batches
            unsup_losses = unsup_losses / self.nb_batches

            real_labels_eval, pred_labels_eval = self._eval(dataloader_valuation, dataset.preprocess, model)
            real_labels_test, pred_labels_test = self._eval(dataloader_test, dataset.preprocess, model)
            metrics_eval = self._get_metrics(real_labels_eval, pred_labels_eval)
            metrics_test = self._get_metrics(real_labels_test, pred_labels_test)

            self._update_graphs(metrics_eval, metrics_test, losses, sup_losses, unsup_losses)

            if epoch % TRAIN_STEP == 0:
                self._update_checkpoint(model, epoch)
                self._save_graphs()

        if epoch % TRAIN_STEP != 0:
            self._update_checkpoint(model, epoch)
            self._save_graphs()

        timer = datetime.now() - start_time

        self._save_train_duration(timer)

    def test(self, dataset, model, trained_model_path, verbose):

        _, _, dataloader_test = dataset.get_dataloaders(self.cuda_state)
        model.load(trained_model_path)

        list_classification_reports = []
        dict_metrics_scores = {}

        if verbose:
            pbar = tqdm(range(TEST_RUNS))
        else:
            pbar = range(TEST_RUNS)

        for i in pbar:
            real_labels, pred_labels = self._eval(dataloader_test, dataset.preprocess, model)
            list_classification_reports.append(classification_report(real_labels, pred_labels, digits=3, output_dict=True))
            metrics = self._get_metrics(real_labels, pred_labels)
            for key in METRICS.keys():
                if key not in dict_metrics_scores.keys():
                    dict_metrics_scores[key] = []
                dict_metrics_scores[key].append(metrics[key])

        full_classification_report = avg_classification_reports(list_classification_reports)
        for key in dict_metrics_scores.keys():
            dict_metrics_scores[key] = np.mean(dict_metrics_scores[key])

        if verbose:
            print(full_classification_report)
        with open(os.path.join(trained_model_path, 'results.txt'), 'a+') as f:
            full_classification_report = f'Number of test runs: {TEST_RUNS}\n' + full_classification_report + '\n'
            f.write(full_classification_report)
            f.write(get_metrics_report(dict_metrics_scores))

    def get_info(self):

        infos = f'Method: {self.name}\n'
        infos += self._get_hyperparameters_info()

        return infos
