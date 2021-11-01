"""
TO DO:
Write a few more display functions
"""

from typing import Sequence

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import torch
from .annotation_reader import *
from .dataset import *
from PIL import Image

LABEL_TO_COLOR = {
    "without_mask": "red",
    "with_mask": "green",
    "mask_weared_incorrect": "blue",
}

INDEX_TO_COLOR = list(LABEL_TO_COLOR.values())


def display_annotation(dataset: FaceMaskDetectionDataset, idx: int) -> None:

    switch_type = False

    if dataset.dataset_type == "segmentation":
        switch_type = True
        dataset.dataset_type = "object_detection"

    img, annotations = dataset[idx]

    fig, ax = plt.subplots()
    ax.imshow(img.permute(1, 2, 0))
    ax.axis("off")

    for idx in range(len(annotations["boxes"])):

        boxes = annotations["boxes"][idx]

        xmin = boxes[0]
        ymin = boxes[1]
        xmax = boxes[2]
        ymax = boxes[3]

        rect = patches.Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            edgecolor=INDEX_TO_COLOR[annotations["labels"][idx] - 1],
            facecolor="none",
        )
        ax.add_patch(rect)

    plt.tight_layout()
    plt.show()

    if switch_type:
        dataset.dataset_type = "segmentation"


def display_augmented_samples(dataset: FaceMaskDetectionDataset, idx: Sequence[int]):

    if isinstance(idx, int):
        idx = [idx]

    for count, i in enumerate(idx):

        augmented_img = dataset[i][0]
        reference_img = dataset.images[dataset.keys[i]]

        if not dataset.load_to_RAM:
            reference_img = reference_img.open()

        plt.subplot(len(idx), 2, (count * 2) + 1)
        plt.axis("off")
        plt.imshow(reference_img.permute(1, 2, 0))
        plt.subplot(len(idx), 2, (count * 2) + 2)
        plt.axis("off")
        plt.imshow(augmented_img.permute(1, 2, 0))

    plt.tight_layout()
    plt.show()


def display_inferences(
    model: torch.nn.Module,
    dataset: FaceMaskDetectionDataset,
    idx: Sequence[int],
    device: torch.device = None,
    threshold=0.5,
) -> None:

    COLORS_BY_CLASS = {0: [0, 0, 0], 1: [255, 0, 0], 2: [0, 255, 0], 3: [0, 0, 255]}

    if isinstance(idx, int):
        idx = [idx]

    if device == None:
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    model.to(device)
    model.eval()

    for count, i in enumerate(idx):

        img = dataset[i][0]
        inference = model(img.to(device).unsqueeze(0))

        boxes = inference[0]["boxes"].to("cpu").detach()
        labels = inference[0]["labels"].to("cpu").detach()
        scores = inference[0]["scores"].to("cpu").detach()

        fig, ax = plt.subplots()
        ax.imshow(img.permute(1, 2, 0))
        ax.axis("off")

        for idx2 in range(boxes.shape[0]):

            box = boxes[idx2]
            label = labels[idx2]
            score = scores[idx2]

            xmin = box[0]
            ymin = box[1]
            xmax = box[2]
            ymax = box[3]

            rect = patches.Rectangle(
                (xmin, ymin),
                xmax - xmin,
                ymax - ymin,
                edgecolor=INDEX_TO_COLOR[label - 1],
                facecolor="none",
            )
            ax.add_patch(rect)

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":

    train, val, test = get_datasets(
        "../../data", load_to_RAM=False, dataset_type="object_detection"
    )

    display_annotation(train, 341)
