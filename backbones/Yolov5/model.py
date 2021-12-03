# adapted from YOLOv5/detect.py 🚀 by Ultralytics
import torch
import torch.nn as nn
from torchvision.transforms import Resize, Pad
import numpy as np

from yolov5.models.experimental import attempt_load
from yolov5.utils.general import non_max_suppression, scale_coords, check_img_size
from yolov5.utils.augmentations import letterbox

class Yolov5(nn.Module):
    def __init__(self,
                weights,
                conf_thres=0.25,
                iou_thres=0.45,
        ):
        super().__init__()

        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        self.imgsz = 640
        self.stride = 64

        self.model = attempt_load(weights)

        def inference_mode(self, val):
            self.inference_mode = val
            
        self.model.inference_mode = inference_mode.__get__(self.model)
        

    def forward(self, img):
        im0_shape = img.shape[2:]
        img = self.reshape(img)

        pred = self.model(img)
        pred = non_max_suppression(pred[0], self.conf_thres, self.iou_thres, max_det=1000)

        pred_list = []

        for det in pred:
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0_shape).round()
            det[:, -1] += 1

            pred_list.append({'boxes': det[:, :4], 'labels': det[:, -1].int(), 'scores': det[:, 4]})

        return pred_list

    def reshape(self, img):
        shape = img.shape[2:]
        new_shape = (640, 640)

        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

        new_unpad = int(round(shape[0] * r)), int(round(shape[1] * r))
        dw, dh = new_shape[1] - new_unpad[1], new_shape[0] - new_unpad[0]
        dw, dh = np.mod(dw, self.stride), np.mod(dh, self.stride)

        dw /= 2 
        dh /= 2

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

        img = Resize(new_unpad)(img)
        img = Pad((left, top, right, bottom), fill=0.447, padding_mode='constant')(img)

        return img