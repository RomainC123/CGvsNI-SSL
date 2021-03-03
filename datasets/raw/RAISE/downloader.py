import requests
import shutil
import pandas as pd

from tqdm.auto import tqdm

# ------------------------------------------------------------------------------

data_type = 'TIFF'

urls = pd.read_csv('RAISE_all.csv')[data_type]

# ------------------------------------------------------------------------------

errors = []

for url in tqdm(urls):

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
