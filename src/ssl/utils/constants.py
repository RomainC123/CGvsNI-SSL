DATA_NO_LABEL = -1  # Label of the unlabeled data
CG_IMG_MULT = 4 # Number of duplicates of CG imgs in CGvsNI datasets
CGVSNI_DATASETS_IDS = {
    'Artlantis': 1,
    'Autodesk': 2,
    'Corona': 3,
    'VRay': 4
}

VAL_NUMBER = {
    'CIFAR10': 5000,
    'SVHN': 5000,
    'MNIST': 5000,
    'CGvsNI': 200
}
STD = 0.15

TRAIN_STEP = 10
LOG_INTERVAL = 2
DEFAULT_EPOCHS = 300
TEST_RUNS = 3

BATCH_SIZE = {
    'CIFAR10': 100,
    'SVHN': 100,
    'MNIST': 100,
    'CGvsNI': 32
}

DATALOADER_PARAMS_CUDA = {'shuffle': True, 'num_workers': 8, 'pin_memory': True}
DATALOADER_PARAMS_NO_CUDA = {'shuffle': True}
