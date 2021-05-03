import os
import pathlib
import pickle
import pandas as pd

from PIL import Image

from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')

data_to_label = {
    'RAISE': 0,
    'VISION': 0,
    'Artlantis': 1,
    'Autodesk': 2,
    'Corona': 3,
    'VRay': 4
}

ROOT_PATH = pathlib.Path(__file__).resolve().parents[3].absolute()

FRAME_PATH = os.path.join(ROOT_PATH, 'datasets', 'CGvsNI')
if not os.path.exists(FRAME_PATH):
    os.makedirs(FRAME_PATH)

RAW_PATH = os.path.join(FRAME_PATH, 'raw')
if not os.path.exists(RAW_PATH):
    os.makedirs(RAW_PATH)

list_names = []
list_labels = []

for data_type in os.listdir(RAW_PATH):
    imgs_path = os.path.join(RAW_PATH, data_type)
    for img_name in os.listdir(imgs_path):
        list_names.append(os.path.join(data_type, img_name))
        list_labels.append(data_to_label[data_type])

df_data = pd.DataFrame(columns=['Name', 'Label'])
df_data['Name'] = list_names
df_data['Label'] = list_labels

with open(os.path.join(FRAME_PATH, 'dataset.csv'), 'w+') as f:
    f.write(df_data.to_csv(index=None))
