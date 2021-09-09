"""
TO DO:
Weight Freezing
"""


from typing import Tuple

import pytorch_lightning as plight
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation import fcn_resnet50
from torchvision.transforms import Normalize


class FCN_ResNet50(nn.Module):

    def __init__(self,
                 n_classes: int = 4):

        super().__init__()

        self.n_classes = n_classes

        self.input_norm = Normalize([0.4778, 0.4581, 0.4503], [
                                    0.2596, 0.2531, 0.2519])

        self.model = fcn_resnet50(pretrained=True)

        # Freeze weights
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.classifier = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, n_classes, kernel_size=1, padding=0, stride=1)
        )

    def forward(self, x):

        x = self.input_norm(x)
        x = self.model(x)['out']

        return x


class FCN_ResNet50_Lightning(plight.LightningModule):

    def __init__(self, n_classes: int = 4):

        super().__init__()
        self.n_classes = n_classes
        self.model = FCN_ResNet50(n_classes)

    def forward(self, x):

        return self.model(x)

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)

        return optimizer

    def training_step(self, batch, batch_idx):

        img, labels = batch

        pred = self(img)

        loss = F.cross_entropy(pred, labels)

        self.log("Training loss", loss, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):

        img, labels = batch

        pred = self(img)
        pred_ = pred.argmax(1)

        loss = F.cross_entropy(pred, labels)
        val_acc = (pred_ == labels).sum() / labels.numel()

        # Per-class accuracy:
        class_accuracy = torch.zeros((self.n_classes,))
        for c in range(self.n_classes):
            if (labels == c).sum() != 0:
                acc = torch.logical_and(
                    (labels == c), (pred_ == c)).sum() / (labels == c).sum()
                class_accuracy[c] = acc

        class_accuracy = class_accuracy.mean()

        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.log("val_acc", val_acc, on_step=False, on_epoch=True)
        self.log("val_acc_per_class", class_accuracy,
                 on_step=False, on_epoch=True)

        return class_accuracy


if __name__ == '__main__':

    pass
