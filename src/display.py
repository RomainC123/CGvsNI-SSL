################################################################################
#   Libraries                                                                  #
################################################################################

import os
import argparse

from ssl.utils.paths import TRAINED_MODELS_PATH, SAVED_MODELS_PATH
from ssl.utils.tools import show_graphs, show_schedules

################################################################################
#   Argparse                                                                   #
################################################################################


def get_args():

    parser = argparse.ArgumentParser(description='Semi-supervised MNIST display')

    # Folder to use
    parser.add_argument('--trained', dest='trained', action='store_true')
    parser.set_defaults(trained=False)
    parser.add_argument('--saved', dest='saved', action='store_true')
    parser.set_defaults(saved=False)

    # Functionalities
    parser.add_argument('--graphs', dest='graphs', action='store_true')
    parser.set_defaults(graphs=False)
    parser.add_argument('--examples', dest='examples', action='store_true')
    parser.set_defaults(examples=False)
    parser.add_argument('--schedules', dest='schedules', action='store_true')
    parser.set_defaults(schedules=False)

    # Data to use
    parser.add_argument('--model_name', type=str, help='model to show')

    args = parser.parse_args()

    assert(args.trained or args.saved)
    assert(args.graphs or args.examples or args.schedules)
    assert(args.model_name != None)

    return args

################################################################################
#   Displays                                                                   #
################################################################################


def main():

    args = get_args()

    if args.trained:
        main_path = TRAINED_MODELS_PATH
    elif args.saved:
        main_path = SAVED_MODELS_PATH
    model_path = os.path.join(main_path, args.model_name)
    if not os.path.exists(model_path):
        raise RuntimeError('Please provide a valid model name')

    if args.graphs:

        show_graphs(model_path)

    if args.schedules:

        show_schedules()


if __name__ == '__main__':
    main()
