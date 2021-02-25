import requests
import shutil
import pandas as pd

from tqdm.auto import tqdm

# ------------------------------------------------------------------------------

df_Artlantis_full = pd.read_csv('Artlantis_URL.txt', header=None)

df_Artlantis = df_Artlantis_full.iloc[1::2].reset_index(drop=True)
df_Artlantis.columns = ['Name']
df_Artlantis['Url'] = df_Artlantis_full.iloc[::2].reset_index(drop=True)

# ------------------------------------------------------------------------------

errors = []

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
