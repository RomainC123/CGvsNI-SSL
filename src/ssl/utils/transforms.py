import torchvision.transforms as transforms


class TransformTwice:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        return out1, out2


IMAGE_TRANSFORMS_TRAIN = {
    'MNIST': {
        'L': transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize((0.5), (0.5))]),
        'RGB': transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    },
    'CIFAR10': {
        'L': transforms.Compose([transforms.RandomCrop(32, padding=2, padding_mode='reflect'),
                                 transforms.RandomHorizontalFlip(0.5),
                                 transforms.ToTensor()]),
        'RGB': transforms.Compose([transforms.RandomCrop(32, padding=2, padding_mode='reflect'),
                                   transforms.RandomHorizontalFlip(0.5),
                                   transforms.ToTensor()])
    },
    'SVHN': {
        'L': transforms.Compose([transforms.RandomCrop(32, padding=2, padding_mode='reflect'),
                                 transforms.RandomHorizontalFlip(0.5),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5), (0.5))]),
        'RGB': transforms.Compose([transforms.RandomCrop(32, padding=2, padding_mode='reflect'),
                                   transforms.RandomHorizontalFlip(0.5)
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    },
    'CGvsNI': {
        'L': transforms.Compose([transforms.RandomCrop(233),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5), (0.5))]),
        'RGB': transforms.Compose([transforms.RandomCrop(233),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    }
}

IMAGE_TRANSFORMS_TEST = {
    'MNIST': {
        'L': transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize((0.5), (0.5))]),
        'RGB': transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    },
    'CIFAR10': {
        'L': transforms.Compose([transforms.ToTensor()]),
        'RGB': transforms.Compose([transforms.ToTensor()])
    },
    'SVHN': {
        'L': transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize((0.5), (0.5))]),
        'RGB': transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    },
    'CGvsNI': {
        'L': transforms.Compose([transforms.RandomCrop(233),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5), (0.5))]),
        'RGB': transforms.Compose([transforms.RandomCrop(233),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    }
}
