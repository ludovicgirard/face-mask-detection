"""
TO DO:
Write a few more display functions
"""

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import torch
from annotation_reader import *
from PIL import Image

LABEL_TO_COLOR = {"without_mask": 'red',
                  "with_mask": 'green',
                  "mask_weared_incorrect": 'blue'}


def display_annotation(img: torch.Tensor, annotations: AnnotationCollection) -> None:

    fig, ax = plt.subplots()
    ax.imshow(img)

    for annotation in annotations:

        xmin = annotation.x_coords[0]
        ymin = annotation.y_coords[0]
        xmax = annotation.x_coords[1]
        ymax = annotation.y_coords[1]

        rect = patches.Rectangle(
            (xmin, ymin), xmax - xmin, ymax - ymin,
            edgecolor=LABEL_TO_COLOR[annotation.label], facecolor='none')
        ax.add_patch(rect)

    plt.show()


def display_few_samples(data_directory):
    pass


def display_specific_samples(data_directory, difficult=True):
    pass


if __name__ == '__main__':

    file = '../../data/annotations/maksssksksss0.xml'
    img = '../../data/images/maksssksksss0.png'

    annotation = read_PASCAL_VOC_xml_file(file)
    img = Image.open(img)

    display_annotation(img, annotation)
