from ssl.data.image import ImageDatasetContainer

kwargs_train = {'batch_size': 100, 'shuffle': False}
kwargs_test = {'batch_size': 100, 'shuffle': False}


cont = ImageDatasetContainer('CIFAR10', 10000, 2000)

cont.make_dataloaders(kwargs_train, img_mode='RGB')
dataloader_train, dataloader_valuation, dataloader_test = cont.get_dataloaders()
print(cont.get_info())
