import torch
from torch.utils.data import Dataset
import pandas as pd
from pathlib import Path
from skimage import io
from numpy import pi
import os


class OrthographicDataset(Dataset):

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.test_images = []
        self.transform = transform
        self._init_dataset()

    def __len__(self):
        return len(self.test_images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.test_images[idx]
        image = io.imread(img_name)

        if self.transform:
            image = self.transform(image)

        return image

    # loop over subfolders and put all images in a list
    def _init_dataset(self):

        # loop over subfolders chair, table and monitor
        for folders in os.listdir(self.root_dir):
            category_folder = os.path.join(self.root_dir, folders)

            # append each image in the subfolder to the test_image list
            for images in os.listdir(category_folder):
                images_filepath = os.path.join(category_folder, images)
                print(images_filepath)
                self.test_images.append(images_filepath)

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample['image']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image)}