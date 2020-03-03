from __future__ import print_function, division
import time
import datetime
import argparse
from pathlib import Path
import numpy as np
from cornell_dataset import CornellDataset, ToTensor, Normalize, de_normalize


def parse_arguments():
    parser = argparse.ArgumentParser(description='Grasping detection system')
    parser.add_argument('--src_path', type=str,
                        help='Path to the src folder with all the orthographic images.')
    parser.add_argument('--output_path', type=str,
                        help='Path to the folder where all the final images will be saved.')
    args = parser.parse_args()
    return args.src_path, args.output_path


if __name__ == '__main__':
    SRC_PATH, OUTPUT_PATH = parse_arguments()
    if SRC_PATH is None:
        SRC_PATH = Path.cwd() / 'ortographic_modelnet10_dataset_gray_images'
        print(f"No path was given. Using {SRC_PATH} as src path for the images.")
    else:
        SRC_PATH = Path(SRC_PATH)
    if OUTPUT_PATH is None:
        OUTPUT_PATH = Path.cwd() / "result_img"
        print(f"No path was given. Using {OUTPUT_PATH} as src path where all the final images will be saved.")
    else:
        OUTPUT_PATH = Path(OUTPUT_PATH)
    # Make sure output exists
    if not OUTPUT_PATH.exists():
        Path.mkdir(OUTPUT_PATH, parents=True)

    # Orth images data loader
    # TODO

    # Create model
    # TODO
    # print(model.model)

    # For each image, plot the predicted rectangle and save it to OUTPUT_PATH
    # TODO

    print("End of testing orthogonal projection images. Byeeee :D!")
