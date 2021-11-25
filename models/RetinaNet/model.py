from typing import Sequence, Tuple

import pytorch_lightning as plight
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection import retinanet_resnet50_fpn
from torchvision.models.detection.retinanet import RetinaNetHead
from torchvision.transforms import Normalize
from torchvision.ops import nms
from .utils import precompute_mIOU


class RetinaNet(nn.Module):
    def __init__(
        self,
        n_classes: int = 4,
        pretrained: bool = True,
        trainable_backbone_layers: int = 5,
        nms_threshold: float = 0.5,
        score_threshold: float = 0.05,
    ):

        super().__init__()

        self.n_classes = n_classes
        self.pretrained = pretrained
        self.trainable_backbone_layers = trainable_backbone_layers
        self.nms_threshold = nms_threshold
        self.apply_nms = False

        self.input_norm = Normalize([0.4778, 0.4581, 0.4503], [0.2596, 0.2531, 0.2519])

        self.model = retinanet_resnet50_fpn(
            pretrained=self.pretrained,
            trainable_backbone_layers=trainable_backbone_layers,
        )

        self.model.score_thresh = score_threshold

        # Overwrite classification head
        n_anchors = self.model.anchor_generator.num_anchors_per_location()[0]
        self.model.head = RetinaNetHead(
            self.model.backbone.out_channels, n_anchors, self.n_classes
        )

    def forward(self, x, *args):

        x = self.input_norm(x)

        if self.training:  # In training mode, labels are also expected

            x = self.model(x, *args)

        else:

            x = self.model(x)

        if self.apply_nms:

            for idx in range(len(x)):

                x_ = x[idx]
                keep = nms(x_["boxes"], x_["scores"], self.nms_threshold)

                x[idx]["boxes"] = x[idx]["boxes"][keep]
                x[idx]["labels"] = x[idx]["labels"][keep]
                x[idx]["scores"] = x[idx]["scores"][keep]

        return x

    def inference_mode(self, value: bool = True):

        self.apply_nms = value

    def set_score_threshold(self, value):

        self.model.score_thresh = value


class RetinaNetLightning(plight.LightningModule):
    def __init__(
        self,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        classification_loss_weight: float = 1.0,
        regression_loss_weight: float = 1.0,
        scheduler_patience=5,
        **args,
    ):

        super().__init__()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.model = RetinaNet(**args)
        self.classification_loss_weight = classification_loss_weight
        self.regression_loss_weight = regression_loss_weight
        self.scheduler_patience = scheduler_patience

    def forward(self, x, *args):

        return self.model(x, *args)

    def configure_optimizers(self):

        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, patience=self.scheduler_patience, mode="max"
            ),
            "monitor": "mIOU",
        }

        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):

        img, labels = batch

        pred = self(img, labels)

        class_loss = pred["classification"]
        regression_loss = pred["bbox_regression"]

        self.log("Classification loss", class_loss, on_step=False, on_epoch=True)
        self.log("Regression loss", regression_loss, on_step=False, on_epoch=True)

        return (
            self.classification_loss_weight * class_loss
            + self.regression_loss_weight * regression_loss
        )

    def validation_step(self, batch, batch_idx):

        img, labels = batch

        pred = self(img)

        partial_iou = precompute_mIOU(img, pred, labels, n_classes=self.model.n_classes)

        return partial_iou

    def validation_epoch_end(self, batch_parts):

        intersection = torch.zeros((self.model.n_classes))
        union = torch.zeros((self.model.n_classes))

        for batch_result in batch_parts:

            intersection = intersection + batch_result[:, 0]
            union = union + batch_result[:, 1]

        iou = intersection[1:] / union[1:]

        for c in range(0, self.model.n_classes - 1):
            self.log("Class {} IOU".format(c + 1), iou[c])

        miou = iou.mean()  # Exclude background class?

        if torch.isnan(miou):
            miou = 0

        self.log("mIOU", miou)

    def test_step(self, batch, batch_idx):

        img, labels = batch

        pred = self(img)

        pred_ = pred.argmax(1)

        loss = F.cross_entropy(pred, labels)
        val_acc = (pred_ == labels).sum() / labels.numel()

        # Per-class accuracy:
        class_iou = torch.zeros((self.model.n_classes, 2))

        for c in range(self.model.n_classes):
            class_iou[c, 0] = torch.logical_and((labels == c), (pred_ == c)).sum()

            class_iou[c, 1] = torch.logical_or((labels == c), (pred_ == c)).sum()

        return class_iou

    def on_validation_start(self):

        self.model.inference_mode(True)

    def on_validation_end(self):

        self.model.inference_mode(False)


if __name__ == "__main__":

    model = retinanet_resnet50_fpn()

    for param in model.named_parameters():
        print(param[0])
        print(param[1].shape)
