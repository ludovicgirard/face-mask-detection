from copy import deepcopy

import torch
import torch.nn.functional as F


class RandomCrop:
    def __init__(self, shape):

        self.shape = shape

    def __call__(self, img, labels):

        if isinstance(labels, torch.Tensor):
            labels_type = "segmentation"
        elif isinstance(labels, dict):
            labels_type = "object_detection"

        # Pad if needed
        padx, pady = 0, 0
        needs_padding = False
        if img.shape[-2] < self.shape[0]:
            padx = int((self.shape[0] - img.shape[-2]) // 2) + 1
            needs_padding = True
        if img.shape[-1] < self.shape[1]:
            pady = int((self.shape[1] - img.shape[-1]) // 2) + 1
            needs_padding = True
        if needs_padding:
            img = F.pad(img, (pady, pady, padx, padx), "constant", 0)
            if labels_type == "segmentation":
                labels = F.pad(labels, (pady, pady, padx, padx), "constant", 0)
            elif labels_type == "object_detection":
                pass

        # Crop

        x_start = torch.randint(0, high=img.shape[-2] - self.shape[0] + 1, size=(1,))
        y_start = torch.randint(0, high=img.shape[-1] - self.shape[1] + 1, size=(1,))

        img = img[
            ..., x_start : x_start + self.shape[0], y_start : y_start + self.shape[1]
        ]
        if labels_type == "segmentation":
            labels_crop = labels[
                ...,
                x_start : x_start + self.shape[0],
                y_start : y_start + self.shape[1],
            ]
        elif labels_type == "object_detection":
            labels_crop = deepcopy(labels)
            for boxes in labels_crop["boxes"]:
                boxes[1] = boxes[1] + (padx - x_start)
                boxes[0] = boxes[0] + (pady - y_start)
                boxes[3] = boxes[3] + (padx - x_start)
                boxes[2] = boxes[2] + (pady - y_start)

        return img, labels_crop


if __name__ == "__main__":

    img = torch.rand((2, 3, 123, 126))
    labels = torch.rand((2, 3, 123, 126))

    pad = RandomCrop((256, 256))

    img, labels = pad(img, labels)
    print(img.shape)
    print(labels.shape)
