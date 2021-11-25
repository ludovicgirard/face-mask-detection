"""
Adapted from Salma Bendaoud's code
"""

import torch
import torch.nn as nn
import torch.functional as F
import torchvision.models as models
from torchvision.ops import nms
import pytorch_lightning as plight

from .utils import precompute_mIOU


def get_model(output_shape, **args):
    model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True, **args)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, output_shape
    )

    return model


class Faster_RCNN(nn.Module):
    def __init__(self, n_classes=4, nms_threshold=0.5, score_thresh=0.05):

        super().__init__()

        self.n_classes = n_classes
        self.model = get_model(n_classes, box_score_thresh=score_thresh)
        self.apply_nms = False
        self.nms_threshold = nms_threshold

    def forward(self, x, *args):

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

    def inference_mode(self, value):

        self.apply_nms = value


class Faster_RCNN_Lightning(plight.LightningModule):
    def __init__(self, learning_rate=0.001, momentum=0.9, weight_decay=0.0001, **args):

        super().__init__()
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.model = Faster_RCNN(**args)

    def forward(self, x, *args):

        return self.model(x, *args)

    def configure_optimizers(self):

        return torch.optim.SGD(
            self.model.parameters(),
            lr=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )

    def training_step(self, batch, batch_idx):

        img, labels = batch

        pred = self(img, labels)

        loss = sum(loss for loss in pred.values())

        return loss

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
