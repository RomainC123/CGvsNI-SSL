################################################################################
#   Libraries                                                                  #
################################################################################

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import itertools

from sklearn.metrics import classification_report
from torch.utils.data.sampler import Sampler

from .schedules import *
from .constants import DEFAULT_EPOCHS

################################################################################
#   Utils                                                                      #
################################################################################


def save_info(model_path, dataset, model, optimizer, method, verbose):

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    if verbose:
        print(dataset.get_info())
        print('------------------------------------\n' + model.get_info())
        print('------------------------------------\n' + optimizer.get_info())
        print('------------------------------------\n' + method.get_info())
    with open(os.path.join(model_path, 'info.txt'), 'a+') as f:
        f.write(dataset.get_info())
        f.write('\n------------------------------------\n' + model.get_info())
        f.write('\n------------------------------------\n' + optimizer.get_info())
        f.write('\n------------------------------------\n' + method.get_info())


def get_latest_log(model_path):

    list_logs = os.listdir(os.path.join(model_path, 'logs'))

    latest_log_id = 0
    latest_log_epoch = 0
    try:
        latest_log = list_logs[0]
    except:
        raise RuntimeError('No logs were found')

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


def show_graphs(model_path):

    graphs_path = os.path.join(model_path, 'graphs')

    with open(os.path.join(graphs_path, 'metrics_eval.pkl'), 'rb') as f:
        metrics_eval = pickle.load(f)
    with open(os.path.join(graphs_path, 'metrics_test.pkl'), 'rb') as f:
        metrics_test = pickle.load(f)
    with open(os.path.join(graphs_path, 'loss.pkl'), 'rb') as f:
        losses = pickle.load(f)
    with open(os.path.join(graphs_path, 'sup_loss.pkl'), 'rb') as f:
        sup_losses = pickle.load(f)
    with open(os.path.join(graphs_path, 'unsup_loss.pkl'), 'rb') as f:
        unsup_losses = pickle.load(f)

    epochs = len(losses)

    fig = plt.figure(figsize=(14, 12))

    ax1 = fig.add_subplot(211)
    ax1.set_title('Metrics')
    for key in metrics_eval.keys():
        ax1.plot(range(epochs), metrics_eval[key], label=key.capitalize() + ' eval')
    for key in metrics_test.keys():
        ax1.plot(range(epochs), metrics_test[key], label=key.capitalize() + ' test')
    ax1.legend()

    ax2 = fig.add_subplot(212)
    ax2.set_title('Loss')
    ax2.plot(range(epochs), losses, label='Total loss')
    ax2.plot(range(epochs), sup_losses, label='Supervised loss')
    ax2.plot(range(epochs), unsup_losses, label='Unsupervised loss')
    ax2.legend()

    plt.show()


def show_schedules(start_epoch=1, total_epochs=DEFAULT_EPOCHS):

    fig = plt.figure(figsize=(12, 24))
    ax1 = fig.add_subplot(311)
    ax1.set_title('Unsupervised weight schedule')
    list_unsup_weight = []
    for epoch in range(start_epoch, total_epochs + 1):
        list_unsup_weight.append(UNSUP_WEIGHT_SCHEDULE(epoch, total_epochs))
    ax1.plot(range(total_epochs - start_epoch + 1), list_unsup_weight)

    ax1 = fig.add_subplot(312)
    ax1.set_title('Learning rate schedule')
    list_lr = []
    for epoch in range(start_epoch, total_epochs + 1):
        list_lr.append(LR_SCHEDULE(epoch, total_epochs))
    ax1.plot(range(total_epochs - start_epoch + 1), list_lr)

    ax1 = fig.add_subplot(313)
    ax1.set_title('Beta 1 schedule')
    list_beta1 = []
    for epoch in range(start_epoch, total_epochs + 1):
        list_beta1.append(B1_SCHEDULE(epoch, total_epochs))
    ax1.plot(range(total_epochs - start_epoch + 1), list_beta1)

    plt.show()

# TODO


def show_example(args):

    if args.data in DATASETS_IMPLEMENTED.keys():
        test_dataset_transforms = TEST_TRANSFORMS[args.img_mode]
        test_dataset = DATASETS_IMPLEMENTED[args.data](args.data,
                                                       args.dataset_name,
                                                       args.img_mode,
                                                       'Test',
                                                       transform=test_dataset_transforms)
        model = MODELS[args.data]
        logs_path = os.path.join(args.trained_model_path, 'logs')
        latest_log, epoch = utils.get_latest_log(logs_path)
        checkpoint = torch.load(os.path.join(logs_path, latest_log))
        model.load_state_dict(checkpoint['state_dict'])
    else:
        raise RuntimeError('Data type not implemented')

    if args.cuda:
        model = model.cuda()

    nb_imgs_to_show = NB_IMGS_TO_SHOW

    fig = plt.figure(figsize=(12, 12))

    id_to_show = random.sample(range(len(test_dataset)), nb_imgs_to_show)
    subplot_id = 1

    for i in id_to_show:

        img, target = test_dataset[i]
        if args.cuda:
            img = img.cuda()

        result = model(torch.unsqueeze(img, 0))
        result = F.softmax(result, dim=1)
        pred_label = result.data.max(1, keepdim=True)[1]

        img = img.cpu().numpy()
        if img.shape[1] == img.shape[2]:
            img = np.transpose(img, (1, 2, 0))  # Edge case can be annoying
        if np.amin(img) < 0:
            img = ((img / 2 + 0.5) * 255).astype(np.uint8)

        ax = fig.add_subplot(3, 3, subplot_id)
        if args.img_mode == 'L':
            ax.imshow(img, cmap='gray_r')
        elif args.img_mode == 'RGB':
            ax.imshow(img)
        ax.set_title(f'Prediction/True label: {pred_label.squeeze().cpu().numpy()}/{target}')
        ax.axis('off')

        subplot_id += 1

    plt.show()


def hter(y_true, y_pred):

    if len(set(y_true)) > 2:
        raise ValueError('Too many classes for HTER')

    count_false_accepts = 0
    count_false_rejects = 0
    count_accepts = 0
    count_rejects = 0

    for i in range(len(y_true)):
        if y_pred[i] == 0:
            count_rejects += 1
            if y_true[i] != y_pred[i]:
                count_false_rejects += 1
        elif y_pred[i] == 1:
            count_accepts += 1
            if y_true[i] != y_pred[i]:
                count_false_accepts += 1

    far = 0.5 * (count_false_accepts / count_accepts + count_false_rejects / count_rejects)

    return far

def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())

def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)

class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size
        self.batch_size = batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                   grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size
