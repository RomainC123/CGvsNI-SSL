import os
import pathlib
import pickle
import random
import numpy as np

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

ROOT_PATH = os.path.join(pathlib.Path(__file__).parent.absolute())  # Src files path


def show_loss(args):

    with open(os.path.join(args.graphs_path, 'loss.pkl'), 'rb') as f:
        losses = pickle.load(f)

    with open(os.path.join(args.graphs_path, 'sup_loss.pkl'), 'rb') as f:
        sup_losses = pickle.load(f)

    with open(os.path.join(args.graphs_path, 'unsup_loss.pkl'), 'rb') as f:
        unsup_losses = pickle.load(f)

    fig = plt.figure(figsize=(12, 24))

    ax1 = fig.add_subplot(311)
    ax1.set_title('Loss')
    ax1.plot(range(args.epochs), losses)

    ax2 = fig.add_subplot(312)
    ax2.set_title('Supervised Loss')
    ax2.plot(range(args.epochs), sup_losses)

    ax3 = fig.add_subplot(313)
    ax3.set_title('Unsupervised Loss')
    ax3.plot(range(args.epochs), unsup_losses)

    plt.show()


def testing_display(test_dataloader, model, args):

    nb_imgs_to_check = 9
    fig = plt.figure(figsize=(12, 12))

    id_to_check = random.sample(range(args.nb_img_test), nb_imgs_to_check)
    subplot_id = 1

    for i in id_to_check:

        img, target = test_dataloader.dataset[i]
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
