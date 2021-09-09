import torch
from PIL import Image
from torchvision.transforms import ToTensor

TO_TENSOR = ToTensor()


class FaceMaskImage:

    def __init__(self, path: str):

        self.path = path

    def open(self) -> torch.Tensor:

        img = TO_TENSOR(Image.open(self.path))

        return img
