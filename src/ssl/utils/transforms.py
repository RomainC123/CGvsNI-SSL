import torchvision.transforms as transforms

IMAGE_TRANSFORMS_TRAIN = {
    'L': transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize((0.5), (0.5))]),
    'RGB': transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    'CGvsNI': None
}

IMAGE_TRANSFORMS_TEST={
    'L': transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize((0.5), (0.5))]),
    'RGB': transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    'CGvsNI': None
}
