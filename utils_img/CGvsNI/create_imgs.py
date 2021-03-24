import os
import pathlib
import argparse
import pandas as pd

import Artlantis.get_images as ArtDL
import Corona.get_images as CorDL

ROOT_PATH = pathlib.Path(__file__).resolve().parents[2].absolute()
FSOURCE_PATH = os.path.join(ROOT_PATH, 'utils', 'CGvsNI')

DATA_PATH = os.path.join(ROOT_PATH, 'datasets', 'GCvsNI')
if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)

parser = argparse.ArgumentParser(description='Downloading tool for CGvsNI dataset')

parser.add_argument('--artlantis', dest='Artlantis', action='store_true')
parser.add_argument('--no-artlantis', dest='Artlantis', action='store_false')
parser.set_defaults(Artlantis=True)
parser.add_argument('--autodesk', dest='Autodesk', action='store_true')
parser.add_argument('--no-autodesk', dest='Autodesk', action='store_false')
parser.set_defaults(Autodesk=True)
parser.add_argument('--corona', dest='Corona', action='store_true')
parser.add_argument('--no-corona', dest='Corona', action='store_false')
parser.set_defaults(Corona=True)
parser.add_argument('--raise', dest='RAISE', action='store_true')
parser.add_argument('--no-raise', dest='RAISE', action='store_false')
parser.set_defaults(RAISE=True)
parser.add_argument('--vision', dest='VISION', action='store_true')
parser.add_argument('--no-vision', dest='VISION', action='store_false')
parser.set_defaults(VISION=True)
parser.add_argument('--vray', dest='Vray', action='store_true')
parser.add_argument('--no-vray', dest='Vray', action='store_false')
parser.set_defaults(Vray=True)

args = parser.parse_args()

if args.Artlantis:
    print('Downloading Artlantis...')
    artlantis_raw_path = os.path.join(DATA_PATH, 'Artlantis', 'raw')
    artlantis_fsource_path = os.path.join(FSOURCE_PATH, 'Artlantis')
    ArtDL.downloader(artlantis_raw_path, artlantis_fsource_path)
    print('Cropping Artlantis...')
    # ArtDL.img_crop(artlantis_raw_path)

if args.Autodesk:
    pass

if args.Corona:
    print('Downloading Corona...')
    corona_raw_path = os.path.join(DATA_PATH, 'Corona', 'raw')
    corona_fsource_path = os.path.join(FSOURCE_PATH, 'Corona')
    CorDL.downloader(corona_raw_path, corona_fsource_path)
    print('Cropping Corona...')
    # CorDL.img_crop(corona_raw_path)

if args.RAISE:
    pass

if args.VISION:
    pass

if args.Vray:
    pass
