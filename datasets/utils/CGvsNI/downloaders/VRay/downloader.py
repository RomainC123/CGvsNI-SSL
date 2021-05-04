import requests
import shutil
import pandas as pd

from tqdm.auto import tqdm

# ------------------------------------------------------------------------------

df_VRay_full = pd.read_csv('VRay_gallery_URL.txt', header=None)

df_VRay = df_VRay_full.iloc[1::2].reset_index(drop=True)
df_VRay.columns = ['Name']
df_VRay['Url'] = df_VRay_full.iloc[::2].reset_index(drop=True)

# ------------------------------------------------------------------------------

with open('errors.txt', 'w') as f:
    for img_data in tqdm(df_VRay.values):

        output_path = 'data/' + img_data[0]
        url = img_data[1]

        try:
            r = requests.get(url, stream=True)

            if r.status_code // 100 == 2:
                r.raw.decode_content = True

                with open(output_path,'wb') as f_img:
                    shutil.copyfileobj(r.raw, f_img)

            else:
                f.write(' '.join(img_data) + ' else ' + str(r.status_code) + '\n')

        except:
            f.write(' '.join(img_data) + ' except ' + str(r.status_code) + '\n')
