import os
import pathlib
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

ROOT_PATH = pathlib.Path(__file__).resolve().parents[1].absolute()

RAW_PATH = os.path.join(ROOT_PATH, 'datasets', 'CIFAR10', 'raw')
if not os.path.exists(RAW_PATH):
    os.makedirs(RAW_PATH)

FRAME_PATH = os.path.join(ROOT_PATH, 'datasets', 'CIFAR10')
if not os.path.exists(FRAME_PATH):
    os.makedirs(FRAME_PATH)

df_imgs = pd.read_csv(os.path.join(FRAME_PATH, 'dataset.csv'))

for img in df_imgs.loc[df_imgs['Label'] == 0][:2].values:
    img_path = os.path.join(RAW_PATH, img[0])
    img_open = Image.open(img_path)
    print(img_open)
    print(np.asarray(img_open))
