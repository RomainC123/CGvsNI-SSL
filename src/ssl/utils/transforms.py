import torchvision.transforms as transforms

IMAGE_TRANSFORMS_TRAIN = {
    'MNIST': {
        'L': [transforms.ToTensor(),
              transforms.Normalize((0.5), (0.5))],
        'RGB': [transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    },
    'CIFAR10': {
        'L': [transforms.ToTensor()],
        'RGB': [transforms.ToTensor()]
    },
    'CGvsNI': {
        'L': [transforms.RandomCrop(233),
              transforms.RandomHorizontalFlip(),
              transforms.ToTensor(),
              transforms.Normalize((0.5), (0.5))],
        'RGB': [transforms.RandomCrop(233),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    }
}

IMAGE_TRANSFORMS_TEST = {
    'MNIST': {
        'L': [transforms.ToTensor(),
              transforms.Normalize((0.5), (0.5))],
        'RGB': [transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    },
    'CIFAR10': {
        'L': [transforms.ToTensor()],
        'RGB': [transforms.ToTensor()]
    },
    'CGvsNI': {
        'L': [transforms.RandomCrop(233),
              transforms.RandomHorizontalFlip(),
              transforms.ToTensor(),
              transforms.Normalize((0.5), (0.5))],
        'RGB': [transforms.RandomCrop(233),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    }
}
