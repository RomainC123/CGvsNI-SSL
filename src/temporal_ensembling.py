################################################################################
#   Libraries                                                                  #
################################################################################

import os
import pathlib
import pickle
import numpy as np

import criterions
import networks

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn import metrics

from tqdm import tqdm


################################################################################
#   Training                                                                   #
################################################################################


def training(train_dataloader, model, optimizer, verbose, args):
    """
    Training function for temporal ensembling
    Takes in a dataloader, a model to train and an optimizer
    Trains the model
    Saves it in logs/
    """

    def get_weight(epoch, args):  # TODO

        percent_labeled = args.percent_labeled

        max_weight_corr = args.hyperparameters['max_weight'] * percent_labeled

        if epoch == 1:
            return 0
        elif epoch >= args.hyperparameters['ramp_epochs']:
            return max_weight_corr
        else:
            return max_weight_corr * np.exp(-args.hyperparameters['ramp_mult'] * (1 - epoch / args.hyperparameters['ramp_epochs']) ** 2)

    def update_moving_average(output, y_ema, epoch, args):

        new_y_ema = torch.zeros(y_ema.shape).float()

        if args.cuda:
            new_y_ema = new_y_ema.cuda()

        for idx in range(len(y_ema)):
            new_y_ema[idx] = (args.hyperparameters['alpha'] * y_ema[idx] + (1 - args.hyperparameters['alpha']) * output[idx]) / (1 - args.hyperparameters['alpha'] ** epoch)

        return new_y_ema

    def train(train_dataloader, model, y_ema, optimizer, criterion, weight_unsupervised_loss, epoch, cuda):

        model.train()

        loss_epoch = torch.tensor([0.], requires_grad=False)
        sup_loss_epoch = torch.tensor([0.], requires_grad=False)
        unsup_loss_epoch = torch.tensor([0.], requires_grad=False)

        if args.cuda:
            loss_epoch = loss_epoch.cuda()
            sup_loss_epoch = sup_loss_epoch.cuda()
            unsup_loss_epoch = unsup_loss_epoch.cuda()

        outputs = torch.zeros(args.nb_img_train, args.nb_classes).float()
        w = torch.autograd.Variable(torch.FloatTensor([weight_unsupervised_loss]), requires_grad=False)

        if cuda:
            outputs = outputs.cuda()
            w = w.cuda()

        if verbose:
            pbar = tqdm(enumerate(train_dataloader))
        else:
            pbar = enumerate(train_dataloader)

        for batch_idx, (data, target) in pbar:

            if cuda:
                data, target = data.cuda(), target.cuda()

            optimizer.zero_grad()

            prediction = model.forward(data)
            y_ema_batch = Variable(y_ema[batch_idx * args.batch_size: (batch_idx + 1) * args.batch_size], requires_grad=False)
            loss, sup_loss, unsup_loss = criterion(prediction, y_ema_batch, target, w)

            outputs[batch_idx * args.batch_size: (batch_idx + 1) * args.batch_size] = prediction.data.clone()

            loss.backward()
            optimizer.step()

            loss_epoch += loss.detach()
            sup_loss_epoch += sup_loss.detach()
            unsup_loss_epoch += unsup_loss.detach()

            if batch_idx % args.log_interval == 0 and verbose:
                pbar.set_description('Train Epoch: {}/{} [{}/{} ({:.0f}%)]. Loss: {:.8f}'.format(epoch,
                                                                                                 args.epochs,
                                                                                                 batch_idx * len(data),
                                                                                                 args.nb_img_train,
                                                                                                 100. * batch_idx / args.nb_batches,
                                                                                                 (loss_epoch / (batch_idx + 1)).item()))

            if batch_idx + 1 >= args.nb_batches and verbose:
                pbar.set_description('Train Epoch: {}/{} [{}/{} ({:.0f}%)]. Loss: {:.8f}'.format(epoch,
                                                                                                 args.epochs,
                                                                                                 args.nb_img_train,
                                                                                                 args.nb_img_train,
                                                                                                 100.,
                                                                                                 (loss_epoch / args.nb_batches).item()))

        if epoch % args.TRAIN_STEP == 0:
            torch.save({'epoch': epoch,
                        'state_dict': model.state_dict()},
                       os.path.join(args.logs_path_temp, f'checkpoint_{epoch}.pth'))

        return outputs, loss_epoch, sup_loss_epoch, unsup_loss_epoch

    # Initialize the model weights and print its layout
    networks.init_weights(model, verbose, init_type='normal')
    if verbose:
        networks.print_network(model)

    # Initialize the temporal moving average for each target
    y_ema = torch.zeros(args.nb_img_train, args.nb_classes).float()

    if args.cuda:
        model.cuda()
        y_ema = y_ema.cuda()

    # First model checkpoint
    torch.save({'epoch': 0,
                'state_dict': model.state_dict()},
               os.path.join(args.logs_path_temp, 'checkpoint_0.pth'))

    # Criterion for calculating the loss of our model
    criterion = criterions.TemporalLoss(args.cuda)

    # Keeping track of each epoch losses
    losses = []
    sup_losses = []
    unsup_losses = []

    for epoch in range(1, args.epochs + 1):

        weight_unsupervised_loss = get_weight(epoch, args)
        output, loss, sup_loss, unsup_loss = train(train_dataloader, model, y_ema, optimizer, criterion, weight_unsupervised_loss, epoch, args.cuda)

        losses.append(loss / int(args.nb_img_train / args.batch_size))
        sup_losses.append(sup_loss / int(args.nb_img_train / args.batch_size))
        unsup_losses.append(unsup_loss / int(args.nb_img_train / args.batch_size))

        y_ema = update_moving_average(output, y_ema, epoch, args)

    with open(os.path.join(args.graphs_path_temp, 'loss.pkl'), 'wb') as f:
        pickle.dump(losses, f)
    with open(os.path.join(args.graphs_path_temp, 'sup_loss.pkl'), 'wb') as f:
        pickle.dump(sup_losses, f)
    with open(os.path.join(args.graphs_path_temp, 'unsup_loss.pkl'), 'wb') as f:
        pickle.dump(unsup_losses, f)

################################################################################
#   Testing                                                                    #
################################################################################


def testing(test_dataloader, model, verbose, args):
    """
    Takes in a dataloader, a trained model
    Returns the prediction over the test set and the target labels over that same set
    """

    # Switch to cuda
    if args.cuda:
        model.cuda()

    model.eval()

    real_labels = []
    pred_labels = []

    pbar = tqdm(enumerate(test_dataloader))

    with torch.no_grad():
        for batch_idx, (data, target) in pbar:

            if args.cuda:
                data, target = data.cuda(), target.cuda()

            # Make the prediciton using the already trained model
            bs, c, h, w = data.size()
            result = model(data.view(-1, c, h, w))
            result = F.softmax(result, dim=1)
            pred = result.data.max(1, keepdim=True)[1]

            # Grab the predictions and the labels into arrays
            real_labels.extend(target.data.cpu().numpy())
            pred_labels.extend(pred.squeeze().cpu().numpy())

            if batch_idx % args.log_interval == 0:
                pbar.set_description('Test Batch: {}/{} [{}/{} ({:.0f}%)]'.format(batch_idx + 1,
                                                                                  args.nb_batches_test,
                                                                                  batch_idx * len(data),
                                                                                  args.nb_img_test,
                                                                                  100. * batch_idx / args.nb_batches_test))
            if batch_idx + 1 >= args.nb_batches_test:
                pbar.set_description('Test Batch: {}/{} [{}/{} ({:.0f}%)]'.format(args.nb_batches_test,
                                                                                  args.nb_batches_test,
                                                                                  args.nb_img_test,
                                                                                  args.nb_img_test,
                                                                                  100.))
    return pred_labels, real_labels
