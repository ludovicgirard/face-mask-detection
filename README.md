# face-mask-detection

## Installation

```
git clone https://github.com/ludovicgirard/face-mask-detection
pip3 install -r face-mask-detection/requirements.txt
```
By default, only the trained MobileNet V3 and YOLO weights are installed.

## Usage

```
python3 face-mask-detection/main.py -w
```

The following arguments and flags are available:
- `-w` or `--webcam` indicates to read the video directly from the webcam. This option has priority over `--video`
- `-v [path]` or `--video [path]` indicates the path to a video to read. Either `-w` or `-v` must be present.
- `-o [path]` or `--output [path]` indicates the path to the output video. If absent, the video is displayed but not written to disk.
- `-b [backbone]` or `--backbone [backbone]` can be one of : MobileNet, RetinaNet, Faster_RCNN and YOLO. Default: YOLO. Case insensitive.
- `-d [device]` or `--device [device]` can be one of: CPU or CUDA. Default: CUDA if available, else CPU.
- `-r [int]` or `--resolution [int]` resolution reduction factor. Implemented to try and reduce compute time, but turns out it doesn't do much. Default : 1.

## Models

| **Model** | **mAP@0.5** | **mAP@0.75** | **mAP for the *no mask* class"**|**mAP for the *mask* class"**|**mAP for the *mask worn incorrectly* class**| **Average frames per second**|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| [RetinaNet](https://arxiv.org/abs/1708.02002) w/ ResNet50 | 0.6545 | 0.4823 |0.3780 |0.6085 |0.2725 | 23.3 |
| [Faster R-CNN](https://arxiv.org/abs/1506.01497) w/ ResNet50 | 0.6799 |0.4865 |0.4509 |0.6063 |0.2022 | 22.1 |
| [Faster R-CNN](https://arxiv.org/abs/1506.01497) w/ MobileNet V3 |0.5643| 0.2993 |0.2951 |0.5228 | 0.0555 | 73.0 |
| [YOLO](https://arxiv.org/abs/1506.02640)| **0.9188** | **0.8127** | **0.6664** | **0.7416** | **0.7187** |**187.5**|

## Dataset

The models were trained, validated and tested on :
https://www.kaggle.com/andrewmvd/face-mask-detection
