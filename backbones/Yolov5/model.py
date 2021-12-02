# adapted from YOLOv5/detect.py 🚀 by Ultralytics
import torch
import torch.nn as nn
from torchvision.transforms import Resize, Normalize

from yolov5.models.experimental import attempt_load
from yolov5.utils.general import non_max_suppression, scale_coords

class Yolov5(nn.Module):
    def __init__(self,
                weights,
                conf_thres=0.25,  # confidence threshold
                iou_thres=0.45,
        ):
        super().__init__()

        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        self.reshape = Resize((640, 640))

        self.model = attempt_load(weights)

        def inference_mode(self, val):
            self.inference_mode = val
            
        self.model.inference_mode = inference_mode.__get__(self.model)
        

    def forward(self, img):
        im0_shape = img.shape[2:]
        img = self.reshape(img)

        pred = self.model(img)[0]
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres)

        pred_list = []

        for det in pred:
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0_shape).round()
            det[:, -1] += 1

            pred_list.append({'boxes': det[:, :4], 'labels': det[:, -1].int(), 'scores': det[:, 4]})

        return pred_list