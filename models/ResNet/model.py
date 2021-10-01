"""
TODO:
When computing weighted validation accuracy, consider not all classes are
present in the sample
"""


from typing import Sequence, Tuple

import pytorch_lightning as plight
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation import fcn_resnet50
from torchvision.transforms import Normalize


class FCN_ResNet50(nn.Module):

    def __init__(self,
                 n_classes: int = 4,
                 freeze: bool = True,
                 pretrained: bool = True):

        super().__init__()

        self.n_classes = n_classes
        self.freeze = freeze
        self.pretrained = pretrained

        self.input_norm = Normalize([0.4778, 0.4581, 0.4503], [
                                    0.2596, 0.2531, 0.2519])

        self.model = fcn_resnet50(pretrained=self.pretrained)

        # Freeze weights
        if self.freeze:
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

    def __init__(self, n_classes: int = 4,
                 freeze: bool = True,
                 learning_rate: float = 1e-4,
                 pretrained: bool = True,
                 weight: Sequence[float] = [1, 1, 1, 1],
                 milestones: Sequence[int] = None):

        super().__init__()
        self.n_classes = n_classes
        self.freeze = freeze
        self.learning_rate = learning_rate
        self.pretrained = pretrained
        self.weight = nn.Parameter(torch.Tensor(weight), requires_grad=False)
        self.model = FCN_ResNet50(n_classes, freeze, pretrained)
        self.milestones = milestones

    def forward(self, x):

        return self.model(x)

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        _return = optimizer
        if self.milestones is not None:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, self.milestones, 0.1)

            _return = [optimizer], [scheduler]

        return _return

    def training_step(self, batch, batch_idx):

        img, labels = batch

        pred = self(img)

        loss = F.cross_entropy(pred, labels, weight=self.weight)

        self.log("Training loss", loss, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):

        img, labels = batch

        pred = self(img)
        pred_ = pred.argmax(1)

        loss = F.cross_entropy(pred, labels)
        val_acc = (pred_ == labels).sum() / labels.numel()

        # Per-class accuracy:
        class_iou = torch.zeros((self.n_classes,))
        for c in range(self.n_classes):
            if (labels == c).sum() != 0:
                iou = torch.logical_and(
                    (labels == c), (pred_ == c)).sum() / torch.logical_or(
                    (labels == c), (pred_ == c)).sum()

                self.log("class_{}_iou".format(c), iou,
                         on_step=False, on_epoch=True)

                class_iou[c] = iou

        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.log("val_acc", val_acc, on_step=False, on_epoch=True)

        return loss


if __name__ == '__main__':

    pass
