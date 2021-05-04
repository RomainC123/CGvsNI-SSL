import os
import pathlib
import pickle
import pandas as pd

from PIL import Image
from tensorflow.keras.datasets import cifar10

from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')

ROOT_PATH = pathlib.Path(__file__).resolve().parents[3].absolute()

RAW_PATH = os.path.join(ROOT_PATH, 'datasets', 'CIFAR10', 'raw')
if not os.path.exists(RAW_PATH):
    os.makedirs(RAW_PATH)

FRAME_PATH = os.path.join(ROOT_PATH, 'datasets', 'CIFAR10')
if not os.path.exists(FRAME_PATH):
    os.makedirs(FRAME_PATH)

print('Downloading CIFAR...')

try:
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
except:
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

list_names = []
list_labels = []

print('Creating images...')

nb_imgs_train = len(X_train)
nb_imgs_test = len(X_test)

for i in tqdm(range(nb_imgs_train)):
    im = Image.fromarray(X_train[i])
    im = im.convert("RGB")
    im.save(os.path.join(RAW_PATH, f'cifar10_{i}.jpeg'))
    list_names.append(f'cifar10_{i}.jpeg')
    list_labels.append(int(y_train[i][0]))

for i in tqdm(range(nb_imgs_test)):
    im = Image.fromarray(X_test[i])
    im = im.convert("RGB")
    im.save(os.path.join(RAW_PATH, f'cifar10_{i + nb_imgs_train}.jpeg'))
    list_names.append(f'cifar10_{i + nb_imgs_train}.jpeg')
    list_labels.append(int(y_test[i][0]))

print('Creating dataframe...')
df_imgs = pd.DataFrame(columns=['Name', 'Label'])
df_imgs['Name'] = list_names
df_imgs['Label'] = list_labels

with open(os.path.join(FRAME_PATH, 'dataset.csv'), 'w+') as f:
    f.write(df_imgs.to_csv(index=None))
