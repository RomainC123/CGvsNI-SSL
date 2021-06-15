import os
import pathlib
import pickle
import requests
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

with requests.get('ufldl.stanford.edu/housenumbers/train.tar.gz', allow_redirects=True) as r:
    open('train_32x32.mat', 'wb').write(r.content)

with requests.get('ufldl.stanford.edu/housenumbers/test.tar.gz', allow_redirects=True) as r:
    open('test_32x32.mat', 'wb').write(r.content)

data_train = loadmat('train_32x32.mat')
data_test = loadmat('test_32x32.mat')

X_train = data_train['X']
y_train = data_train['y']
X_test = data_test['X']
y_test = data_test['y']

list_names = []
list_labels = []

print('Creating images...')

nb_imgs_train = X_train.shape[3]
nb_imgs_test = X_test.shape[3]

for i in tqdm(range(nb_imgs_train)):
    im = Image.fromarray(X_train[:,:,:,i])
    im = im.convert("RGB")
    im.save(os.path.join(RAW_PATH, f'svhn_{i}.jpeg'))
    list_names.append(f'svhn_{i}.jpeg')
    label = int(y_train[i])
    if label == 10:
        list_labels.append(0)
    else:
        list_labels.append(label)

for i in tqdm(range(nb_imgs_test)):
    im = Image.fromarray(X_test[:,:,:,i])
    im = im.convert("RGB")
    im.save(os.path.join(RAW_PATH, f'svhn_{i + nb_imgs_train}.jpeg'))
    list_names.append(f'svhn_{i + nb_imgs_train}.jpeg')
    label = int(y_train[i])
    if label == 10:
        list_labels.append(0)
    else:
        list_labels.append(label)

print('Creating dataframe...')
df_imgs = pd.DataFrame(columns=['Name', 'Label'])
df_imgs['Name'] = list_names
df_imgs['Label'] = list_labels

print(df_imgs['Label'].value_counts())

with open(os.path.join(FRAME_PATH, 'dataset.csv'), 'w+') as f:
    f.write(df_imgs.to_csv(index=None))

os.remove('train_32x32.mat')
os.remove('test_32x32.mat')
