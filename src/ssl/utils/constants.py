TRAIN_STEP = 10
LOG_INTERVAL = 2

DATA_NO_LABEL = -1  # Label of the unlabeled data
VAL_NUMBER = 5000
VAL_NUMBER_CGVSNI = 100
TEST_RUNS = 3
ONLY_SUP_RUNS = 10
NB_IMGS_TO_SHOW = 9
BATCH_SIZE = 50
STD = 0.15

DEFAULT_EPOCHS = 300

DATALOADER_PARAMS_CUDA = {'batch_size': BATCH_SIZE, 'shuffle': False, 'num_workers': 8, 'pin_memory': True}
DATALOADER_PARAMS_NO_CUDA = {'batch_size': BATCH_SIZE, 'shuffle': False}

CGVSNI_DATASETS_IDS = {
    'Artlantis': 1,
    'Autodesk': 2,
    'Corona': 3,
    'VRay': 4
}
