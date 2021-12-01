# face-mask-detection

## Installation

```
git clone https://github.com/ludovicgirard/face-mask-detection
pip3 install -r face-mask-detection/requirements.txt
```
By default, only the trained MobileNet V3 weights are installed.

## Usage

```
python3 face-mask-detection/main.py -w
```

The following arguments and flags are available:
- `-w` or `--webcam` indicates to read the video directly from the webcam. This option has priority over `--video`
- `-v [path]` or `--video [path]` indicates the path to a video to read. Either `-w` or `-v` must be present.
- `-o [path]` or `--output [path]` indicates the path to the output video. If absent, the video is displayed but not written to disk.
- `-b [backbone]` or `--backbone [backbone]` can be one of : MobileNet, RetinaNet or Faster_RCNN. Default: MobileNet. Case insensitive.
- `-d [device]` or `--device [device]` can be one of: CPU or CUDA. Default: CUDA if available, else CPU.
- `-r [int]` or `--resolution [int]` resolution reduction factor. Implemented to try and reduce compute time, but turns out it doesn't do much. Default : 1.

## Models

| **Model** | **mAP@0.5** | **Average frames per second**|
|:---:|:---:|:---:|
| [RetinaNet](https://arxiv.org/abs/1708.02002) w/ ResNet50 | 0.6545 | 23.3 |
| [Faster R-CNN](https://arxiv.org/abs/1506.01497) w/ ResNet50 | **0.6799** | 22.1 |
| [Faster R-CNN](https://arxiv.org/abs/1506.01497) w/ MobileNet V3 |0.5643| **73.0** |

## Dataset

The models were trained, validated and tested on :
https://www.kaggle.com/andrewmvd/face-mask-detection
