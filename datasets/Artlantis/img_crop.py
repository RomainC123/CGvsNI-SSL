import math
from PIL import Image
import numbers
import os

from tqdm.auto import tqdm

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

kImageDirRoot = '/nethome/caplierr/code/gipsa/datasets/Artlantis'
kSrcDir = os.path.join(kImageDirRoot, 'data')
kDesDir = os.path.join(kImageDirRoot, 'data_512crop')

if not os.path.exists(kDesDir):
    os.makedirs(kDesDir)

file_list = os.listdir(kSrcDir)
print(len(file_list))

for image_name in tqdm(file_list):
    with open(os.path.join(kSrcDir, image_name), 'rb') as f:
        img = Image.open(f)
        img = img.convert('RGB')

    img_r = resize(img, 512, Image.BICUBIC)
    [name, ext] = os.path.splitext(image_name)

    img_r.save(os.path.join(kDesDir, name + '.png'))
