
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


def make_dataset(fpath, mode):

    pass


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

    def __init__(self, fpath, mode, transform=None, target_transform=None, loader=default_loader):

        classes, class_to_idx = find_classes()
        imgs, num_in_class, images_txt = make_dataset(args.dataroot, class_to_idx)

        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + fpath + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.imgs = imgs
        self.num_in_class = num_in_class
        self.images_txt = images_txt
        self.classes = classes
        self.class_to_idx = class_to_idx
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

        path, target = self.imgs[index]
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
