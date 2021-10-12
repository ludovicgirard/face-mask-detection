"""
TO DO:
Implement other geometric transforms (rotation/flip)
"""

import json
import os
from copy import deepcopy
from typing import Callable, Sequence, Tuple

import torch
from annotation_reader import *
from image_reader import *
from torch.utils.data import Subset
from torchvision.transforms import ColorJitter, Compose, GaussianBlur, RandomEqualize
from transforms import RandomCrop

DATASET_MEAN = [0.4778, 0.4581, 0.4503]
DATASET_STD = [0.2596, 0.2531, 0.2519]

TRAINING_TRANSFORMS = Compose(
    [ColorJitter(), GaussianBlur(kernel_size=3), RandomEqualize()]
)

TRAINING_CROP = RandomCrop((256, 256))


class FaceMaskDetectionDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        keys: Sequence[str],
        images: dict,
        annotations: dict,
        transforms: Callable = None,
        load_to_RAM: bool = False,
        crop: Callable = None,
    ):

        self.keys = keys
        self.images = images
        self.annotations = annotations
        self.load_to_RAM = load_to_RAM
        self.transforms = transforms
        self.crop = crop

        self.paths = {
            "keys": deepcopy(self.keys),
            "images": deepcopy(self.images),
            "annotations": deepcopy(self.annotations),
        }

        if self.load_to_RAM:

            for key in self.keys:
                self.images[key] = self.images[key].open()
                self.annotations[key] = self.annotations[key].to_tensor()

    @staticmethod
    def load_from_directory(data_directory: str, **args):

        keys = []
        images = {}
        annotations = {}

        image_directory = os.path.join(data_directory, "images")
        annotation_directory = os.path.join(data_directory, "annotations")

        # Check if all paths are valid
        if not os.path.isdir(image_directory) or not os.path.isdir(
            annotation_directory
        ):
            raise ValueError("Directory does not exist")

        for file in os.listdir(image_directory):

            if ".png" in file:

                key = file.replace(".png", "")
                keys.append(key)

                annotation_file = key + ".xml"

                images[key] = FaceMaskImage(os.path.join(image_directory, file))
                annotations[key] = read_PASCAL_VOC_annotations(
                    os.path.join(annotation_directory, annotation_file)
                )

        return FaceMaskDetectionDataset(keys, images, annotations, **args)

    @staticmethod
    def load_from_json(file: str, **args):

        with open(file, "r") as open_file:

            paths = json.load(open_file)

        keys = paths["keys"]
        images = paths["images"]
        annotations = paths["annotations"]

        for key in keys:
            images[key] = FaceMaskImage(images[key]["path"])
            annotations[key] = read_PASCAL_VOC_annotations(annotations[key]["filename"])

        return FaceMaskDetectionDataset(keys, images, annotations, **args)

    def save_to_json(self, output_file: str):

        with open(output_file, "w") as file:

            json.dump(self.paths, file, default=lambda x: x.__dict__, indent=4)

    def __len__(self) -> int:

        return len(self.keys)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor]:

        key = self.keys[index]

        image = self.images[key]
        label = self.annotations[key]

        if not self.load_to_RAM:
            image = image.open()
            label = label.to_tensor()

        if self.transforms is not None:

            image = self.transforms((image * 255.0).to(torch.uint8)) / 255.0

        if self.crop is not None:

            image, label = self.crop(image, label)

        return image, label


def subset(dataset: FaceMaskDetectionDataset, index: Sequence[int]):

    _dataset = deepcopy(dataset)
    _dataset.keys = [dataset.keys[idx] for idx in index]

    return _dataset


def get_datasets(
    data_directory: str,
    split_sizes: Tuple[float] = (0.6, 0.2, 0.2),
    load_to_RAM: bool = False,
    save_partition=False,
    load_partition=False,
):

    full_dataset = FaceMaskDetectionDataset.load_from_directory(
        data_directory, load_to_RAM=load_to_RAM
    )

    train_set_samples = int(split_sizes[0] * len(full_dataset))
    val_set_samples = train_set_samples + int(split_sizes[1] * len(full_dataset))
    test_set_samples = int(split_sizes[2] * len(full_dataset))

    index = torch.randperm(len(full_dataset)).tolist()

    if isinstance(save_partition, str):
        with open(save_partition, "w") as file:
            json.dump(index, file, indent=4)

    if isinstance(load_partition, str):
        with open(load_partition, "r") as file:
            index = json.load(file)

    train_set = subset(full_dataset, index[:train_set_samples])
    val_set = subset(full_dataset, index[train_set_samples:val_set_samples])
    test_set = subset(full_dataset, index[val_set_samples:])

    del full_dataset

    train_set.transforms = TRAINING_TRANSFORMS
    train_set.crop = TRAINING_CROP

    return train_set, val_set, test_set


def get_loaders(
    data_directory: str,
    split_sizes: Tuple[float] = (0.6, 0.2, 0.2),
    load_to_RAM: bool = False,
    **args
):

    train_set, val_set, test_set = get_datasets(
        data_directory, split_sizes, load_to_RAM, **args
    )

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=1, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, num_workers=4)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":

    directory = "../../data"
    dset = FaceMaskDetectionDataset(directory, load_to_RAM=True)

    # Compute dataset statistics
    total_pixels = 0
    rgb_mean = torch.zeros((3, len(dset)))
    rgb_std = torch.zeros((3, len(dset)))

    for idx in range(len(dset)):

        img, labels = dset[idx]

        print(img.shape)

        img = img.flatten(-2, -1)

        _n_pixels = img.shape[1]
        rgb_mean[:, idx] = img.mean(dim=1) * _n_pixels
        rgb_std[:, idx] = img.std(dim=1) * _n_pixels
        total_pixels += _n_pixels

    rgb_mean = rgb_mean.sum(dim=1) / total_pixels
    rgb_std = rgb_std.sum(dim=1) / total_pixels
