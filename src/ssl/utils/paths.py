import os
import pathlib

ROOT_PATH = pathlib.Path(__file__).resolve().parents[3].absolute()

DATASETS_PATH = os.path.join(ROOT_PATH, 'datasets')  # dataset.csv files path
if not os.path.exists(DATASETS_PATH):
    raise RuntimeError('No dataset folder found, please create it')

TRAINED_MODELS_PATH = os.path.join(ROOT_PATH, 'trained_models')  # Graphs path
if not os.path.exists(TRAINED_MODELS_PATH):
    os.makedirs(TRAINED_MODELS_PATH)

SAVED_MODELS_PATH = os.path.join(ROOT_PATH, 'saved_models')  # Graphs path
if not os.path.exists(SAVED_MODELS_PATH):
    os.makedirs(SAVED_MODELS_PATH)
