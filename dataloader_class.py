import os
import pathlib
import pandas as pd
import torch.utils.data as data

from PIL import Image
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler

ROOT_PATH = os.path.join(pathlib.Path(__file__).parent.absolute(), 'datasets')
DATA_PATH = os.path.join(ROOT_PATH, 'raw')
INPUT_PATH = os.path.join(ROOT_PATH, 'clean')

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG', '.TIF',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

ID_TO_CLASSES = {
    0: 'CG',
    1: 'N',
    'Nan': 'Testing set or no label'
}


def load_dataset(fpath, mode):

    df_imgs = pd.read_csv(fpath)

    if mode == 'train':
        df_imgs = df_imgs.loc[~df_imgs['Test']][['Name', 'Label']].reset_index(drop=True)
    if mode == 'test':
        df_imgs = df_imgs.loc[df_imgs['Test']][['Name']].reset_index(drop=True)

    return df_imgs  # shuffle somewhere ?


def get_info(df_imgs, idx):
    if len(df_imgs.columns) == 2:
        return df_imgs.iloc[idx]['Name'], df_imgs.iloc[idx]['Label']
    else:
        return df_imgs.iloc[idx]['Name'], None


def pil_loader(path, mode='RGB'):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        if mode == 'L':
            return img.convert('L')  # convert image to grey
        elif mode == 'RGB':
            return img.convert('RGB')  # convert image to rgb image
        elif mode == 'HSV':
            return img.convert('HSV')
        # elif mode == 'LAB':
        #     return RGB2Lab(img)


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path, mode='RGB'):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path, mode)


def show_image(dataset_image):
    dataset_image[0].show()


class ImageCGNIDataset(data.Dataset):

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

    def __init__(self, args, train, transform=None, target_transform=None, loader=default_loader):

        fpath = os.path.join(INPUT_PATH, args.dataset_name)

        self.imgs = load_dataset(fpath, train)  # pandas dataframe

        if len(self.imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + fname + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.input_nc = args.input_nc
        self.img_mode = args.img_mode
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __len__(self):
        """
        Overloading the len function
        """

        return len(self.imgs)

    def __getitem__(self, idx):
        """
        Overloading the Dataset[i] syntax
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """

        if idx > len(self.imgs):
            raise ValueError(f'Index out of bounds: {idx}')

        path, target = get_info(self.imgs, idx)

        img = self.loader(os.path.join(DATA_PATH, path))

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class DataLoaderCGNI(DataLoader):
    def __init__(self, dataset, shuffle=False, batch_size=1, drop_last=True, num_workers=0, pin_memory=False):
        if shuffle:
            sampler = RandomSampler(dataset)
        else:
            sampler = SequentialSampler(dataset)

        super(DataLoaderCGNI, self). \
            __init__(dataset, batch_size, None, sampler,
                     None, num_workers, pin_memory=pin_memory, drop_last=drop_last)
