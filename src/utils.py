################################################################################
#   Libraries                                                                  #
################################################################################

import os

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

        f.write(classification_report(args.real_labels, args.pred_labels, digits=3))
        f.write('\n-----------------------------------------------------------------------------')
