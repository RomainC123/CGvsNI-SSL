import torchvision.transforms as transforms

IMAGE_TRANSFORMS_TRAIN = {
    'MNIST': {
        'L': transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize((0.5), (0.5))]),
        'RGB': transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    },
    'CIFAR10': {
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

IMAGE_TRANSFORMS_TEST = {
    'MNIST': {
        'L': transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize((0.5), (0.5))]),
        'RGB': transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    },
    'CIFAR10': {
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
