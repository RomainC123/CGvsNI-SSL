################################################################################
#   Libraries                                                                  #
################################################################################

import os
import pickle
import pandas as pd
import numpy as np

from vars import *

import torch
from sklearn.metrics import classification_report

################################################################################
#   Utils                                                                      #
################################################################################


class WeightSchedule:

    def __init__(self, ramp_up_epochs, ramp_up_mult, ramp_down_epochs=0, ramp_down_mult=0, start_epoch=0):

        self.ramp_up_epochs = ramp_up_epochs
        self.ramp_up_mult = ramp_up_mult
        self.ramp_down_epochs = ramp_down_epochs
        self.ramp_down_mult = ramp_down_mult

        self.weight = 0.
        self.epoch = start_epoch
        self.ramp_down = 0

    def step(self, total_epochs, start_epoch=0):

        self.epoch += 1

        if self.epoch <= self.ramp_up_epochs:
            self.weight = np.exp(-self.ramp_up_mult * (1 - self.epoch / self.ramp_up_epochs) ** 2)
        elif self.epoch >= total_epochs - self.ramp_down_epochs and self.ramp_down_epochs != 0:
            self.weight = np.exp(-self.ramp_down_mult * (self.ramp_down / self.ramp_down_epochs) ** 2)
            self.ramp_down += 1
        else:
            self.weight = 1.

        return self.weight


def get_train_id(fpath):
    """
    Grabs a training id
    """
    list_full_names = [folder_name.split('_') for folder_name in os.listdir(fpath)]
    list_idx_taken = []
    for splitted_name in list_full_names:
        list_idx_taken.append(int(splitted_name[0]))

    idx = 1
    while idx in list_idx_taken:
        idx += 1
    return str(idx)


def get_trained_model_from_id(train_id):

    folder_list = os.listdir(TRAINED_MODELS_PATH)
    list_ids = [folder_name.split('_')[0] for folder_name in folder_list]
    for i in range(len(folder_list)):
        if int(list_ids[i]) == train_id:
            return folder_list[i]
    raise RuntimeError(f'Train id not found: {train_id}')


def get_container_info(model, nb_img_train, nb_classes, percent_labeled, epochs, batch_size, nb_batches, shuffle, method, train_id, optimizer, init_mode):

    def get_optimizer_info(optimizer):
        params_dict = optimizer.state_dict()['param_groups'][0]
        str_optim = f'Optimizer: {type(optimizer).__name__}\n'
        str_optim += f"Learning rate: {params_dict['lr']}\n"
        str_optim += f"Betas: {params_dict['betas']}\n"
        return str_optim

    info_string = f'Train_id: {train_id}\n'
    info_string += f'Model: {type(model).__name__}\n'
    info_string += f'Number of training images: {nb_img_train}\n'
    info_string += f'Number of classes: {nb_classes}\n'
    info_string += 'Percent of labeled samples: {:.2f}\n'.format(percent_labeled * 100)
    info_string += f'Epochs: {epochs}\n'
    info_string += f'Batch size: {batch_size}\n'
    info_string += f'Number of batches: {nb_batches}\n'
    info_string += f'Shuffle train set: {shuffle}\n'
    info_string += method.get_info()
    info_string += get_optimizer_info(optimizer)
    info_string += f'Init mode: {init_mode}\n'

    return info_string


def get_nb_parameters(model):
    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    return f'Trainable paramaters: {num_params}\n'


def get_latest_log(logs_path):

    list_logs = os.listdir(logs_path)

    latest_log_id = 0
    latest_log_epoch = 0
    latest_log = list_logs[0]

    for i in range(len(list_logs)):
        log_epoch = list_logs[i].split('_')[-1].split('.')[0]
        if int(log_epoch) > latest_log_epoch:
            latest_log = list_logs[i]
            latest_log_epoch = int(log_epoch)
            latest_log_id = i

    return latest_log, latest_log_epoch


def update_checkpoint(model, epoch_id, logs_path):

    latest_log, latest_log_epoch_id = get_latest_log(logs_path)
    torch.save({'epoch': epoch_id,
                'state_dict': model.state_dict()},
               os.path.join(logs_path, f'checkpoint_{epoch_id}.pth'))
    os.remove(os.path.join(logs_path, latest_log))


def save_graphs(graphs_path, accuracy, losses, sup_losses, unsup_losses):

    with open(os.path.join(graphs_path, 'accuracy.pkl'), 'wb') as f:
        pickle.dump(accuracy, f)
    with open(os.path.join(graphs_path, 'loss.pkl'), 'wb') as f:
        pickle.dump(losses, f)
    with open(os.path.join(graphs_path, 'sup_loss.pkl'), 'wb') as f:
        pickle.dump(sup_losses, f)
    with open(os.path.join(graphs_path, 'unsup_loss.pkl'), 'wb') as f:
        pickle.dump(unsup_losses, f)


def avg_classifications_reports(list_classification_reports):

    accuracy = 0
    avg_report = {}
    nb_reports = len(list_classification_reports)

    for report in list_classification_reports:
        for key in report.keys():
            if isinstance(report[key], dict):
                if key not in avg_report.keys():
                    avg_report[key] = {}
                for sub_key in report[key].keys():
                    if sub_key == 'support' and sub_key not in avg_report[key].keys():
                        avg_report[key][sub_key] = report[key][sub_key]
                    else:
                        if sub_key not in avg_report[key].keys():
                            avg_report[key][sub_key] = report[key][sub_key]
                        else:
                            avg_report[key][sub_key] += report[key][sub_key]
            if key == 'accuracy':
                accuracy += report[key]

    for key in avg_report.keys():
        if isinstance(avg_report[key], dict):
            for sub_key in avg_report[key]:
                avg_report[key][sub_key] /= nb_reports

    df_report = pd.DataFrame(avg_report).transpose()
    df_report['precision'] = df_report['precision'].apply(lambda x: round(x, 3))
    df_report['recall'] = df_report['recall'].apply(lambda x: round(x, 3))
    df_report['f1-score'] = df_report['f1-score'].apply(lambda x: round(x, 3))
    df_report['support'] = df_report['support'].astype(int)
    report_str = df_report.to_string() + '\naccuracy          {:.3f}'.format(accuracy / nb_reports)

    return report_str


def get_hyperparameters_combinations(method):

    hyperparameters = HYPERPARAMETERS_SEARCH[method]

    keys, values = zip(*hyperparameters.items())
    permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]

    return permutations_dicts


def get_avg_metrics(list_dict_metrics):

    avg_metrics = {}

    for metric_name in METRICS.keys():
        avg_metrics[metric_name] = 0.
        for i in range(ONLY_SUP_RUNS):
            avg_metrics[metric_name] += list_dict_metrics[i][metric_name]
        avg_metrics[metric_name] /= ONLY_SUP_RUNS

    return avg_metrics


def get_metrics_report(dict_metrics):

    report = ''

    for metric_name in dict_metrics.keys():
        report += metric_name.capitalize() + ': {:.3f}\n'.format(dict_metrics[metric_name])

    return report
