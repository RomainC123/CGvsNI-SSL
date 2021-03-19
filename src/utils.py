################################################################################
#   Libraries                                                                  #
################################################################################

import os
import pandas as pd

from vars import *

from sklearn.metrics import classification_report

################################################################################
#   Utils                                                                      #
################################################################################


def get_train_id():
    """
    Grabs a training id
    """
    list_full_names = [folder_name.split('_') for folder_name in os.listdir(TRAINED_MODELS_PATH)]
    list_idx_taken = []
    for splitted_name in list_full_names:
        list_idx_taken.append(int(splitted_name[-1]))

    idx = 1
    while idx in list_idx_taken:
        idx += 1
    return str(idx)


def get_trained_model_from_id(train_id):

    folder_list = os.listdir(TRAINED_MODELS_PATH)
    list_ids = [folder_name.split('_')[-1] for folder_name in folder_list]
    for i in range(len(folder_list)):
        if int(list_ids[i]) == train_id:
            return folder_list[i]
    raise RuntimeError(f'Train id not found: {train_id}')


def get_train_info(nb_img_train, nb_classes, percent_labeled, epochs, batch_size, nb_batches, shuffle, method, train_id, optimizer):

    def get_optimizer_info(optimizer):
        params_dict = optimizer.state_dict()['param_groups'][0]
        str_optim = f'Optimizer: {type(optimizer).__name__}\n'
        str_optim += f"Learning rate: {params_dict['lr']}\n"
        str_optim += f"Betas: {params_dict['betas']}\n"
        return str_optim

    info_string = f'Train_id: {train_id}\n'
    info_string += f'Number of training images: {nb_img_train}\n'
    info_string += f'Number of classes: {nb_classes}\n'
    info_string += 'Percent of labeled samples: {:.1f}\n'.format(percent_labeled * 100)
    info_string += f'Epochs: {epochs}\n'
    info_string += f'Batch size: {batch_size}\n'
    info_string += f'Number of batches: {nb_batches}\n'
    info_string += f'Shuffle train set: {shuffle}\n'
    info_string += method.get_info()
    info_string += get_optimizer_info(optimizer)

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

    return latest_log


def avg_classifications_reports(list_classification_reports):

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
            else:
                if key not in avg_report.keys():
                    avg_report[key] = 0
                avg_report[key] += report[key]

    for key in avg_report.keys():
        if isinstance(avg_report[key], dict):
            for sub_key in avg_report[key]:
                if sub_key != 'support':
                    avg_report[key][sub_key] /= nb_reports
        else:
            avg_report[key] /= nb_reports

    return pd.DataFrame(avg_report).transpose().to_string()


def save_results(param_id, args):

    def save_hyperparams(param_id, args):

        params_str = ''
        params_combi = get_hyperparameters_combinations(args.method)
        params = params_combi[int(param_id) - 1]
        for param in params.keys():
            params_str += param + ': ' + str(params[param]) + '\n'

        return params_str

    spath = os.path.join(args.results_path, 'classification_report.txt')
    if not os.path.exists(spath):
        with open(spath, 'w+') as f:
            f.write(f'Data type:  {args.data}\n')
            f.write(f'Dataset name: {args.dataset_name}\n')
            f.write(f'Train size: {args.nb_train}\n')
            f.write('Percent labeled: {:.1f}%\n'.format(args.percent_labeled * 100))
            f.write(f'Test size: {args.nb_test}\n')
            f.write('\n-----------------------------------------------------------------------------')

    with open(spath, 'a') as f:
        f.write('\n')
        f.write(f'Training method: {args.method}\n')
        f.write(f'Training id: {args.train_id}\n')
        f.write('Method hyperparameters: \n')
        f.write(save_hyperparams(param_id, args))
        f.write('\n')

        f.write(args.full_classification_report)
        f.write('\n-----------------------------------------------------------------------------')
