import os
import pathlib
import pickle
from scipy.io import loadmat  # this is the SciPy module that loads mat-files
import pandas as pd

from PIL import Image
import matplotlib.pyplot as plt


from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')

ROOT_PATH = pathlib.Path(__file__).resolve().parents[3].absolute()

RAW_PATH = os.path.join(ROOT_PATH, 'datasets', 'SVHN', 'raw')
if not os.path.exists(RAW_PATH):
    os.makedirs(RAW_PATH)

FRAME_PATH = os.path.join(ROOT_PATH, 'datasets', 'SVHN')
if not os.path.exists(FRAME_PATH):
    os.makedirs(FRAME_PATH)

data_train = loadmat(os.path.join(FRAME_PATH, 'train_32x32.mat'))
data_test = loadmat(os.path.join(FRAME_PATH, 'test_32x32.mat'))

X_train = data_train['X']
y_train = data_train['y']
X_test = data_test['X']
y_test = data_test['y']

list_names = []
list_labels = []

print('Creating images...')

nb_imgs_train = len(X_train)
nb_imgs_test = len(X_test)

for i in tqdm(range(nb_imgs_train)):
    im = Image.fromarray(X_train[:,:,:,i])
    im = im.convert("RGB")
    im.save(os.path.join(RAW_PATH, f'svhn_{i}.jpeg'))
    list_names.append(f'svhn_{i}.jpeg')
    list_labels.append(int(y_train[i][0]))

for i in tqdm(range(nb_imgs_test)):
    im = Image.fromarray(X_test[:,:,:,i])
    im = im.convert("RGB")
    im.save(os.path.join(RAW_PATH, f'svhn_{i + nb_imgs_train}.jpeg'))
    list_names.append(f'svhn_{i + nb_imgs_train}.jpeg')
    list_labels.append(int(y_test[i][0]))

print('Creating dataframe...')
df_imgs = pd.DataFrame(columns=['Name', 'Label'])
df_imgs['Name'] = list_names
df_imgs['Label'] = list_labels

with open(os.path.join(FRAME_PATH, 'dataset.csv'), 'w+') as f:
    f.write(df_imgs.to_csv(index=None))
