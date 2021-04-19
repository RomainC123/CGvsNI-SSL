import torchvision.transforms as transforms

IMAGE_TRANSFORMS_TRAIN = {
    'L': transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize((0.5), (0.5))]),
    'RGB': transforms.Compose([transforms.RandomCrop(32, padding=2, padding_mode='reflect'),
                                transforms.RandomHorizontalFlip(0.5),
                                transforms.ToTensor()]),
    'RGB_old': transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    'CGvsNI': None
}

IMAGE_TRANSFORMS_TEST={
    'L': transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize((0.5), (0.5))]),
    'RGB': transforms.Compose([transforms.ToTensor()]),
    'RGB_old': transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    'CGvsNI': None
}
