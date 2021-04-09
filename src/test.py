from ssl.datasets.base import ImageDatasetContainer

kwargs_train = {'batch_size': 100, 'shuffle': False}
kwargs_test = {'batch_size': 100, 'shuffle': False}


cont = ImageDatasetContainer('MNIST', 10000, 2000)

dataloaders = cont.get_dataloaders('RGB', kwargs_train)
