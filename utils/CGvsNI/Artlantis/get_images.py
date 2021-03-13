import os
import pathlib
import requests
import pickle
import pandas as pd

from PIL import Image

from tqdm import tqdm

def downloader(artlantis_raw_path, artlantis_fsource_path):

    if not os.path.exists(artlantis_raw_path):
        os.makedirs(artlantis_raw_path)

    if not os.path.exists(artlantis_fsource_path):
        os.makedirs(artlantis_fsource_path)

    df_Artlantis_full = pd.read_csv(os.path.join(artlantis_fsource_path, 'Artlantis_URL.txt'), header=None)

    df_Artlantis = df_Artlantis_full.iloc[1::2].reset_index(drop=True)
    df_Artlantis.columns = ['Name']
    df_Artlantis['Url'] = df_Artlantis_full.iloc[::2].reset_index(drop=True)

    errors = []

    for img_data in tqdm(df_Artlantis.values):
        fpath = os.path.join(artlantis_raw_path, img_data[0])
        url = img_data[1]
        try:
            r = requests.get(url)
            if r.status_code == 200:
                with open(fpath,'wb') as f:
                    f.write(r.content)
        except:
            errors.append(img_data)

    with open(os.path.join(artlantis_fsource_path, 'errors.txt'), 'w+') as f:
        for error in errors:
            f.write(' '.join(error) + '\n')

    print(f'{len(errors)} errors')
