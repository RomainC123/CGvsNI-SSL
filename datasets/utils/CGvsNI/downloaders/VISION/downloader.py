import requests
import shutil
import pandas as pd

from tqdm.auto import tqdm

# ------------------------------------------------------------------------------

df_VISION_all = pd.read_csv('VISON_URL.txt', header=None)
df_VISION_all.columns = ['Url']
df_VISION = df_VISION_all.loc[df_VISION_all['Url'].apply(lambda x: x.split('/')[-1].split('.')[-1] == 'jpg')].reset_index(drop=True)

# ------------------------------------------------------------------------------

errors = []

for url in tqdm(df_VISION['Url']):

    output_path = 'data/' + url.split('/')[-1]

    try:
        r = requests.get(url, stream=True)
    except:
        errors.append(img_data)

    if r.status_code == 200:
        r.raw.decode_content = True

        with open(output_path,'wb') as f:
            shutil.copyfileobj(r.raw, f)
    else:
        errors.append(img_data)

with open('errors.txt', 'w') as f:
    for error in errors:
        f.write(' '.join(error) + '\n')
