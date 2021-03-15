import os
import pathlib
import argparse
import pickle
import numpy as np

import torch
import torch.nn.functional as F

from sklearn import metrics
from display import testing_display

from tqdm import tqdm


ROOT_PATH = pathlib.Path(__file__).resolve().parents[1].absolute()


def testing_metrics(test_dataloader, model, args):

    if args.cuda:
        model.cuda()

    model.eval()

    oriImageLabel = []
    oriTestLabel = []

    pbar = tqdm(enumerate(test_dataloader))

    with torch.no_grad():
        for batch_idx, (data, target) in pbar:

            if args.cuda:
                data, target = data.cuda(), target.cuda()

            bs, c, h, w = data.size()
            result = model(data.view(-1, c, h, w))
            result = F.softmax(result, dim=1)
            pred = result.data.max(1, keepdim=True)[1]
            oriTestLabel.extend(pred.squeeze().cpu().numpy())
            oriImageLabel.extend(target.data.cpu().numpy())

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

    print(metrics.classification_report(oriImageLabel, oriTestLabel, digits=3))
