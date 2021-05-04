import os
import pathlib
import pandas as pd

from PIL import Image
from sklearn.datasets import fetch_openml
from tqdm import tqdm

ROOT_PATH = pathlib.Path(__file__).resolve().parents[3].absolute()

RAW_PATH = os.path.join(ROOT_PATH, 'datasets', 'MNIST', 'raw')
if not os.path.exists(RAW_PATH):
    os.makedirs(RAW_PATH)

FRAME_PATH = os.path.join(ROOT_PATH, 'datasets', 'MNIST')
if not os.path.exists(FRAME_PATH):
    os.makedirs(FRAME_PATH)

print('Downloading MNIST...')

try:
    mnist = fetch_openml('mnist_784', as_frame=True)
except:
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    mnist = fetch_openml('mnist_784', as_frame=True)

X, y = mnist["data"], mnist["target"]
y = pd.DataFrame(y)

list_names = []

print('Creating images...')
for i in tqdm(range(len(X))):
    im = Image.fromarray(X.iloc[i].values.reshape(28, 28))
    im = im.convert("L")
    im.save(os.path.join(RAW_PATH, f"mnist_784_{i}.jpeg"))
    list_names.append(f"mnist_784_{i}.jpeg")

y.insert(0, "Name", list_names, True)
y.rename(columns={'class': 'Label'}, inplace=True)

with open(os.path.join(FRAME_PATH, 'dataset.csv'), 'w+') as f:
    f.write(y.to_csv(index=None))
