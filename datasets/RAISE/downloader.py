import urllib.request
import pandas as pd
import pickle

from tqdm.auto import tqdm

# ------------------------------------------------------------------------------


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

# ------------------------------------------------------------------------------


data_type = 'TIFF'

df_RAISE = pd.read_csv('RAISE_all.csv')

urls = df_RAISE[data_type]
errors = []

for url in tqdm(urls[:3]):

    output_path = 'data/' + url.split('/')[-1]

    try:
        download_url(url, output_path)
    except:
        errors.append(url)

with open('errors.pkl', 'wb') as f:
    pickle.dump(errors, f)
