import os
from typing import Callable, Tuple

import torch
from annotation_reader import *
from image_reader import *


class Dataset(torch.utils.data.Dataset):

    def __init__(self, data_directory: str, transforms: Callable = None,
                 read_to_RAM: bool = False):

        self.data_directory = data_directory
        self.read_to_RAM = read_to_RAM
        self.transforms = transforms

        self.image_directory = os.path.join(self.data_directory, 'images')
        self.annotation_directory = os.path.join(
            self.data_directory, 'annotations')

        # Check if all paths are valid
        if not os.path.isdir(self.image_directory) or \
                not os.path.isdir(self.annotation_directory):
            raise ValueError('Directory does not exist')

        self.explore_data_directory()

    def explore_data_directory(self):

        self.images = []
        self.annotations = []

        for file in os.listdir(self.image_directory):

            if '.png' in file:

                annotation_file = file.replace('.png', '.xml')

                self.images.append(FaceMaskImage(
                    os.path.join(self.image_directory, file)))
                self.annotations.append(read_PASCAL_VOC_annotations(
                    os.path.join(self.annotation_directory, annotation_file)))

        if self.read_to_RAM:

            self.images = [image.open() for image in self.images]
            self.annotations = [annotation.to_tensor()
                                for annotation in self.annotations]

    def __len__(self) -> int:

        return len(self.images)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor]:

        image = self.images[index]
        label = self.annotations[index]

        if not self.read_to_RAM:
            image = image.open()
            label = label.to_tensor()

        if self.transforms is not None:

            image, label = self.transforms(image, label)

        return image, label


if __name__ == '__main__':

    directory = '../../data'
    dset = Dataset(directory, read_to_RAM=True)

    im1, label1 = dset[0]
