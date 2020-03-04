from __future__ import print_function, division
import time
import datetime
import argparse
from pathlib import Path
import numpy as np
from cornell_dataset import CornellDataset, ToTensor, Normalize, de_normalize
from prediction_model import PredictionNet
import torch
from util import plot_image


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
    # TODO: NETWORK_NAME, PRE_TRAINED and MODEL_PATH should be added to parse_arguments()
    NETWORK_NAME = "resnet18"
    PRE_TRAINED = True
    MODEL_PATH = "./results/resnet18_pretrained/our_resnet18_2019-11-10_18h53m.pt"
    model = PredictionNet(dest_path=OUTPUT_PATH, orthogonal_loader=orthogonal_loader, network_name=NETWORK_NAME,
                          pre_trained=PRE_TRAINED)
    path = Path(MODEL_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_model(path, device=device)
    # print(model.model)

    # For each image, plot the predicted rectangle and save it to OUTPUT_PATH
    # TODO
    images, predictions = model.get_prediction(orthogonal_loader)
    for i, batch in enumerate(predictions):
        for j, rect in enumerate(batch):
            image = images[i][j]
            image = de_normalize(image, PRE_TRAINED)
            image = image.numpy().transpose((1, 2, 0))
            # print(f"Predicted rectangles {rect}")
            plot_image(image, rect)

    print("End of testing orthogonal projection images. Byeeee :D!")
