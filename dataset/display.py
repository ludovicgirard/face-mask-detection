"""
TO DO:
Write a few more display functions
"""

from typing import Sequence

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import torch
from annotation_reader import *
from dataset import *
from PIL import Image

LABEL_TO_COLOR = {"without_mask": 'red',
                  "with_mask": 'green',
                  "mask_weared_incorrect": 'blue'}


def display_annotation(dataset: FaceMaskDetectionDataset, idx: int) -> None:

    img = dataset[idx][0].permute(1, 2, 0)
    annotations = dataset.annotations[dataset.keys[idx]]

    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.axis('off')

    for annotation in annotations:

        xmin = annotation.x_coords[0]
        ymin = annotation.y_coords[0]
        xmax = annotation.x_coords[1]
        ymax = annotation.y_coords[1]

        rect = patches.Rectangle(
            (xmin, ymin), xmax - xmin, ymax - ymin,
            edgecolor=LABEL_TO_COLOR[annotation.label], facecolor='none')
        ax.add_patch(rect)

    plt.tight_layout()
    plt.show()


def display_augmented_samples(dataset: FaceMaskDetectionDataset, idx: Sequence[int]):

    if isinstance(idx, int):
        idx = [idx]

    for count, i in enumerate(idx):

        augmented_img = dataset[i][0]
        reference_img = dataset.images[dataset.keys[i]]

        if not dataset.load_to_RAM:
            reference_img = reference_img.open()

        plt.subplot(len(idx), 2, (count * 2) + 1)
        plt.axis('off')
        plt.imshow(reference_img.permute(1, 2, 0))
        plt.subplot(len(idx), 2, (count * 2) + 2)
        plt.axis('off')
        plt.imshow(augmented_img.permute(1, 2, 0))

    plt.tight_layout()
    plt.show()


def display_inferences(model: torch.nn.Module,
                       dataset: FaceMaskDetectionDataset,
                       idx: Sequence[int],
                       device=None):

    if isinstance(idx, int):
        idx = [idx]

    if device == None:
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'

    model.to(device)

    for count, i in enumerate(idx):

        img = dataset[i][0]
        inference = model(img.to(device).unsqueeze(0)).to('cpu').detach()

        plt.subplot(len(idx), 2, (count * 2) + 1)
        plt.axis('off')
        plt.imshow(img.permute(1, 2, 0))
        plt.subplot(len(idx), 2, (count * 2) + 2)
        plt.axis('off')
        plt.imshow(inference.squeeze(), cmap='gist_')
    plt.tight_layout()
    plt.plot()


if __name__ == '__main__':

    train, val, test = get_datasets('../../data', load_to_RAM=True)
    # train = FaceMaskDetectionDataset('../../data', load_to_RAM=True)

    display_augmented_samples(train, 341)
