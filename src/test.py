import os
import pathlib
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from ssl.utils.tools import hter

ROOT_PATH = pathlib.Path(__file__).resolve().parents[1].absolute()

RAW_PATH = os.path.join(ROOT_PATH, 'datasets', 'CIFAR10', 'raw')
if not os.path.exists(RAW_PATH):
    os.makedirs(RAW_PATH)

FRAME_PATH = os.path.join(ROOT_PATH, 'datasets', 'CIFAR10')
if not os.path.exists(FRAME_PATH):
    os.makedirs(FRAME_PATH)

print(hter([0, 1, 1, 0, 0], [0, 1, 1, 0, 0]))
print(hter([0, 1, 1, 0, 0, 0], [0, 1, 1, 0, 0, 1]))
print(hter([0, 1, 1, 0, 0, 1], [0, 1, 1, 0, 0, 0]))

df = pd.DataFrame([0, 1, 1, 1, 2, 3, 4])
print(df.sample(n=18, replace=True))
