import os
import pathlib
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

ROOT_PATH = pathlib.Path(__file__).resolve().parents[1].absolute()

RAW_PATH = os.path.join(ROOT_PATH, 'datasets', 'SVHN', 'raw')
if not os.path.exists(RAW_PATH):
    os.makedirs(RAW_PATH)

FRAME_PATH = os.path.join(ROOT_PATH, 'datasets', 'SVHN')
if not os.path.exists(FRAME_PATH):
    os.makedirs(FRAME_PATH)

df_imgs = pd.read_csv(os.path.join(FRAME_PATH, 'dataset.csv'))
print(df_imgs.loc[df_imgs['Label'] == 10][:2])

for img in df_imgs.loc[df_imgs['Label'] == 10][:2].values:
    print(img)
    im = Image.open(os.path.join(RAW_PATH, img[0]))
    plt.imshow(im)
    plt.show()
