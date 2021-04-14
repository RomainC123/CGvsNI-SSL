################################################################################
#   Libraries                                                                  #
################################################################################

import os
import pickle
import numpy as np
import pandas as pd
import torch

from sklearn.metrics import classification_report

################################################################################
#   Weight schedule                                                            #
################################################################################


class WeightSchedule:

    def __init__(self, ramp_up_epochs, ramp_up_mult, ramp_down_epochs, ramp_down_mult):

        self.ramp_up_epochs = ramp_up_epochs
        self.ramp_up_mult = ramp_up_mult
        self.ramp_down_epochs = ramp_down_epochs
        self.ramp_down_mult = ramp_down_mult

    def __call__(self, epoch, total_epochs):

        if epoch <= self.ramp_up_epochs:
            weight = np.exp(-self.ramp_up_mult * (1 - epoch / self.ramp_up_epochs) ** 2)
        elif epoch >= total_epochs - self.ramp_down_epochs and self.ramp_down_epochs != 0:
            ramp_down = epoch - (total_epochs - self.ramp_down_epochs)
            weight = np.exp(-self.ramp_down_mult * (ramp_down / self.ramp_down_epochs) ** 2)
        else:
            weight = 1.

        return weight

################################################################################
#   Utils                                                                      #
################################################################################


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


def avg_classification_reports(list_classification_reports):

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


def get_metrics_report(dict_metrics):

    report = ''

    for metric_name in dict_metrics.keys():
        report += metric_name.capitalize() + ': {:.3f}\n'.format(dict_metrics[metric_name])

    return report
