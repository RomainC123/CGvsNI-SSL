import sys
sys.path.append('../')

import os
import pathlib
import requests
import pickle
import pandas as pd

from PIL import Image
from imageio import imread

from tqdm import tqdm


def resize(img, size, interpolation=Image.BILINEAR):
    r"""Resize the input PIL Image to the given size.
    Args:
        img (PIL Image): Image to be resized.
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), the output size will be matched to this. If size is an int,
            the smaller edge of the image will be matched to this number maintaing
            the aspect ratio. i.e, if height > width, then image will be rescaled to
            :math:`\left(\text{size} \times \frac{\text{height}}{\text{width}}, \text{size}\right)`
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    Returns:
        PIL Image: Resized image.
    """
    if isinstance(size, int):
        w, h = img.size
        print(w, h)
        if (w <= h and w == size) or (h <= w and h == size):
            return img
        if w < h:
            ow = size
            oh = int(size * h / w)
            return img.resize((ow, oh), interpolation)
        else:
            oh = size
            ow = int(size * w / h)
            return img.resize((ow, oh), interpolation)
    else:
        return img.resize(size[::-1], interpolation)


def crop(img, i, j, h, w):
    """Crop the given PIL Image.
    Args:
        img (PIL Image): Image to be cropped.
        i (int): i in (i,j) i.e coordinates of the upper left corner.
        j (int): j in (i,j) i.e coordinates of the upper left corner.
        h (int): Height of the cropped image.
        w (int): Width of the cropped image.
    Returns:
        PIL Image: Cropped image.
    """
    # if not _is_pil_image(img):
    #     raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    return img.crop((j, i, j + w, i + h))


def center_crop(img, output_size):
    if isinstance(output_size, numbers.Number):
        output_size = (int(output_size), int(output_size))
    w, h = img.size
    th, tw = output_size
    i = int(round((h - th) / 2.))
    j = int(round((w - tw) / 2.))
    return crop(img, i, j, th, tw)


def five_crop(img, size):
    """Crop the given PIL Image into four corners and the central crop.
    .. Note::
        This transform returns a tuple of images and there may be a
        mismatch in the number of inputs and targets your ``Dataset`` returns.
    Args:
       size (sequence or int): Desired output size of the crop. If size is an
           int instead of sequence like (h, w), a square crop (size, size) is
           made.
    Returns:
       tuple: tuple (tl, tr, bl, br, center)
                Corresponding top left, top right, bottom left, bottom right and center crop.
    """
    if isinstance(size, numbers.Number):
        size = (int(size), int(size))
    else:
        assert len(size) == 2, "Please provide only two dimensions (h, w) for size."

    w, h = img.size
    crop_h, crop_w = size
    if crop_w > w or crop_h > h:
        raise ValueError("Requested crop size {} is bigger than input size {}".format(size,
                                                                                      (h, w)))
    tl = img.crop((0, 0, crop_w, crop_h))
    tr = img.crop((w - crop_w, 0, w, crop_h))
    bl = img.crop((0, h - crop_h, crop_w, h))
    br = img.crop((w - crop_w, h - crop_h, w, h))
    center = center_crop(img, (crop_h, crop_w))
    return (tl, tr, bl, br, center)


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
            with requests.get(url) as r:
                if r.status_code == 200:
                    with open(fpath,'wb') as f:
                        f.write(r.content)
            del r

            img = Image.open(fpath)
            img = img.convert('RGB')
            img_r = resize(img, 512, Image.BICUBIC)

            if img_r.height > img_r.width:
                xmin = 0
                ymin = 0
                width = 512
                height = int(img_r.height * 0.9)
                J = img_r.crop((xmin, ymin, xmin + width, ymin + height))  # The crop rectangle, as a (left, upper, right, lower)-tuple.
            else:
                xmin = 0
                ymin = 0
                width = img_r.width
                height = int(img_r.height * 0.9)
                J = img_r.crop((xmin, ymin, xmin + width, ymin + height))

            with open(fpath, 'w') as f:
                J.save(f)
        except:
            print('Error except')
            errors.append(img_data)

    with open(os.path.join(corona_fsource_path, 'errors.txt'), 'w+') as f:
        for error in errors:
            f.write(' '.join(error) + '\n')

    print(f'{len(errors)} errors')
