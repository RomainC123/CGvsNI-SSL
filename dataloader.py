import os
import pathlib
import pandas as pd


INPUT_PATH = os.path.join(os.path.join(pathlib.Path(__file__).parent.absolute(), 'datasets'), 'clean')

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG', '.TIF',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

ID_TO_CLASSES = {
    0: 'no label',
    1: 'CG',
    2: 'N',
    'Nan': 'Testing set'
}


def load_dataset(fpath, mode):

    df_imgs = pd.read_csv(fpath).iloc

    if mode == 'train':
        df_imgs = df_imgs.loc[~df_imgs['Test']]
    if mode == 'test':
        df_imgs = df_imgs.loc[df_imgs['Test']]

    return df_imgs


def get_info(df_imgs, idx):


class ImageCGNIDataset(Dataset):

    """
    Class containing images and corresponding class (either CG, N or no label)
    ------
    Args:
        - fpath: file path of the .csv file containing images attributions
        (train, test, and labels)
        - mode: either 'train' or 'test'
    Attributes:
        -
    """

    def __init__(self, fname, mode, transform=None, target_transform=None, loader=default_loader):

        fpath = os.path.join(INPUT_PATH, fname)

        self.imgs = load_dataset(fpath, mode)  # pandas dataframe

        if len(df_imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + fname + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __get__item(self, idx):
        """
        Overloading the Dataset[i] syntax
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """

        path, target = get_info(self.imgs, idx)

        img = self.loader(path, self.mode)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        """
        Overloading the len function
        """
        pass
