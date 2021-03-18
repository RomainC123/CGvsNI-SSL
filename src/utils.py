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


def save_results(args):

    def save_hyperparams(args):

        params_str = ''
        params = HYPERPARAMETERS[args.method]
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
        f.write(save_hyperparams(args))
        f.write('\n')

        f.write(args.full_classification_report)
        f.write('\n-----------------------------------------------------------------------------')
