"""
TO DO:
Add typing hints
"""

import xml.etree.ElementTree as ET
from typing import List, Tuple

import torch

LABEL_TO_VALUE = {"without_mask": 1, "with_mask": 2, "mask_weared_incorrect": 3}


class Annotation:
    """
    TO DO:
    Implement using dataclasses
    """

    def __init__(
        self,
        label: str,
        x_coords: Tuple[int] = None,
        y_coords: Tuple[int] = None,
        pose: str = None,
        truncated: int = None,
        occluded: int = None,
        difficult: int = None,
    ):

        self.label = label
        self.x_coords = x_coords
        self.y_coords = y_coords
        self.pose = pose
        self.truncated = truncated
        self.occluded = occluded
        self.difficult = difficult


class AnnotationCollection:
    """
    Contains the multiple annotations related to one single image
    """

    def __init__(
        self,
        filename: str,
        img_name: str,
        annotation_list: List[Annotation],
        img_size: Tuple[int],
    ):

        self.filename = filename
        self.img_name = img_name
        self.annotation_list = annotation_list
        self.img_size = img_size

    def __iter__(self):

        return (annotation for annotation in self.annotation_list)

    def __getitem__(self, idx):

        return self.annotation_list[idx]

    def to_tensor(self, onehot=False) -> torch.Tensor:

        label_tensor = torch.zeros(self.img_size)

        for annotation in self.annotation_list:

            value = LABEL_TO_VALUE[annotation.label]

            label_tensor[
                annotation.y_coords[0] : annotation.y_coords[1],
                annotation.x_coords[0] : annotation.x_coords[1],
            ] = value

        return label_tensor.to(torch.long)


def read_PASCAL_VOC_annotations(filename: str) -> AnnotationCollection:
    """
    thanks Stack Overflow :
        https://stackoverflow.com/questions/53317592/reading-pascal-voc-annotations-in-python
    """

    tree = ET.parse(filename)
    root = tree.getroot()
    annotation_list = []

    img_name = root.find("filename").text

    img_size = (int(root.find("size/height").text), int(root.find("size/width").text))

    for boxes in root.iter("object"):

        label = boxes.find("name").text
        pose = boxes.find("pose").text
        truncated = int(boxes.find("truncated").text)
        occluded = int(boxes.find("occluded").text)
        difficult = int(boxes.find("difficult").text)
        x_coords = [
            int(boxes.find("bndbox/xmin").text),
            int(boxes.find("bndbox/xmax").text),
        ]
        y_coords = [
            int(boxes.find("bndbox/ymin").text),
            int(boxes.find("bndbox/ymax").text),
        ]

        annotation_list.append(
            Annotation(label, x_coords, y_coords, pose, truncated, occluded, difficult)
        )

    return AnnotationCollection(filename, img_name, annotation_list, img_size)


if __name__ == "__main__":

    file = "../../data/annotations/maksssksksss0.xml"
    a = read_PASCAL_VOC_annotations(file)
    a.to_tensor()
