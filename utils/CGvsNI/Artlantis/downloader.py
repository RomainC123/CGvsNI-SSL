import os
import pathlib
import requests
import pickle
import pandas as pd

from PIL import Image

from tqdm import tqdm

ROOT_PATH = pathlib.Path(__file__).resolve().parents[2].absolute()

RAW_PATH = os.path.join(ROOT_PATH, 'datasets', 'Artlantis', 'raw')
if not os.path.exists(RAW_PATH):
    os.makedirs(RAW_PATH)

df_Artlantis_full = pd.read_csv('Artlantis_URL.txt', header=None)

df_Artlantis = df_Artlantis_full.iloc[1::2].reset_index(drop=True)
df_Artlantis.columns = ['Name']
df_Artlantis['Url'] = df_Artlantis_full.iloc[::2].reset_index(drop=True)

errors = []

print('Downloading Artlantis images...')
for img_data in tqdm(df_Artlantis.values):

    output_path = 'data/' + img_data[0]
    url = img_data[1]

    try:
        r = requests.get(url, stream=True)

        if r.status_code == 200:
            r.raw.decode_content = True

            with open(output_path,'wb') as f:
                shutil.copyfileobj(r.raw, f)

    except:
        errors.append(img_data)

print(f'{len(errors)} errors')
with open('errors.txt', 'w') as f:
    for error in errors:
        f.write(' '.join(error) + '\n')
