import os
import pathlib
import requests
import pickle
import pandas as pd

from PIL import Image

from tqdm import tqdm

def downloader(corona_raw_path, corona_fsource_path):

    if not os.path.exists(corona_raw_path):
        os.makedirs(corona_raw_path)

    if not os.path.exists(corona_fsource_path):
        os.makedirs(corona_fsource_path)

    df_Corona_full = pd.read_csv(os.path.join(corona_fsource_path, 'Corona_URL.txt'), header=None)

    df_Corona = df_Corona_full.iloc[1::2].reset_index(drop=True)
    df_Corona.columns = ['Name']
    df_Corona['Url'] = df_Corona_full.iloc[::2].reset_index(drop=True)

    errors = []

    for img_data in tqdm(df_Corona.values):
        fpath = os.path.join(corona_raw_path, img_data[0])
        url = img_data[1]
        try:
            r = requests.get(url)
            if r.status_code == 200:
                with open(fpath,'wb') as f:
                    f.write(r.content)
        except:
            errors.append(img_data)

    

    with open(os.path.join(corona_fsource_path, 'errors.txt'), 'w+') as f:
        for error in errors:
            f.write(' '.join(error) + '\n')

    print(f'{len(errors)} errors')
