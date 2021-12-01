import cv2
import torch
import argparse
import time
import warnings
import os
import sys

PARENT = os.path.dirname(__file__)
sys.path.append(PARENT)

import models

LABEL_TO_COLOR = {
    "without_mask": "red",
    "with_mask": "green",
    "mask_weared_incorrect": "blue",
}

INDEX_TO_COLOR = [None, (0, 0, 255), (0, 255, 0), (255, 0, 0)]


def parse_args():

    AUTO_SELECT_MODEL = False
    AUTO_SELECT_DEVICE = False

    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--webcam", action="store_true")
    parser.add_argument("-v", "--video", action="store")
    parser.add_argument("-o", "--output", action="store")
    parser.add_argument(
        "-b",
        "--backbone",
        action="store",
        help="Options are : MobileNet, Faster_RCNN, RetinaNet",
    )
    parser.add_argument("-d", "--device", action="store")
    parser.add_argument("-r", "--resolution", action="store", type=int, default=1)

    args = parser.parse_args()

    # DEVICE

    if args.device is None:

        DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

        print(f"Device {DEVICE} was selected automatically")

    elif args.device.lower() == "cuda":

        if torch.cuda.is_available():

            DEVICE = "cuda:0"

        else:

            print(f"Requested device {args.device} is unavailable. Defaulting to CPU")

    elif args.device.lower() == "cpu":

        DEVICE = "cpu"

    # BACKBONE

    if args.backbone is None:
        # Auto select the backbone based on hardware
        AUTO_SELECT_MODEL = True

    elif args.backbone.lower() == "faster_rcnn":
        MODEL = models.Faster_RCNN_Lightning(
            nms_threshold=0.2, score_thresh=0.4, pretrained=False
        )
        MODEL.load_state_dict(
            torch.load(os.path.join(PARENT, "weights/faster_rcnn_lightning.pt"))
        )

    elif args.backbone.lower() == "mobilenet":
        MODEL = models.MobileNet_Lightning(
            nms_threshold=0.2, score_thresh=0.4, pretrained=False
        )
        MODEL.load_state_dict(
            torch.load(os.path.join(PARENT, "weights/faster_rcnn_mobilenet.pt"))
        )

    elif args.backbone.lower() == "retinanet":
        MODEL = models.RetinaNetLightning(nms_threshold=0.2, pretrained=False)
        MODEL.load_state_dict(torch.load(os.path.join(PARENT, "weights/retinanet.pt")))
        MODEL.model.set_score_threshold(0.3)

    else:
        print(
            f"Unrecognized backbone {args.backbone}, selecting backbone based on hardware"
        )
        AUTO_SELECT_MODEL = True

    if AUTO_SELECT_MODEL:

        MODEL = models.MobileNet_Lightning(nms_threshold=0.2, score_thresh=0.4)
        MODEL.load_state_dict(
            torch.load(os.path.join(PARENT, "weights/faster_rcnn_mobilenet.pt"))
        )
        print("Model MobileNet V3 was selected automatically")

    MODEL.eval()
    MODEL.model.inference_mode(True)

    # TREAT INPUT VIDEO

    if args.webcam:
        args.video = -1

    elif not args.webcam and args.video is None:

        raise ValueError("No video input given.")

    return MODEL, DEVICE, args.video, args.output, args.resolution


def display_video_feed(model, device, video, output, downsample):

    model.to(device)

    cam = cv2.VideoCapture(video)
    fps = cam.get(cv2.CAP_PROP_FPS)
    size = (int(cam.get(3)), int(cam.get(4)))

    if output is not None:
        writer = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    else:
        writer = None

    frametime = 1 / fps  # [s]

    while True:

        frame_timer = time.perf_counter()
        ret_val, img = cam.read()

        if not ret_val:
            # cv2.destroyAllWindows()
            break

        img = cv2.flip(img, 1)

        img_tensor = (
            torch.tensor(img).permute(2, 0, 1).unsqueeze(0).to(device, torch.float)
            / 255
        )

        if downsample > 1:
            img_tensor = torch.nn.functional.avg_pool2d(img_tensor, downsample)

        inference = model(img_tensor)

        display_inference(img, inference, downsample, frame_timer, writer)

        current_frametime = time.perf_counter() - frame_timer

        if current_frametime < frametime:
            time.sleep(frametime - current_frametime)

        if cv2.waitKey(1) == 27:
            break  # esc to quit

    cam.release()
    writer.release()
    cv2.destroyAllWindows()


def display_inference(img, inference, downsample, frame_timer, writer):

    boxes = inference[0]["boxes"].to("cpu").detach()
    labels = inference[0]["labels"].to("cpu").detach()
    scores = inference[0]["scores"].to("cpu").detach()

    for idx2 in range(boxes.shape[0]):

        box = boxes[idx2]
        label = labels[idx2]
        score = scores[idx2]

        xmin, ymin, xmax, ymax = map(int, box)

        cv2.rectangle(
            img,
            (xmin * downsample, ymin * downsample),
            (xmax * downsample, ymax * downsample),
            INDEX_TO_COLOR[label],
            thickness=3,
        )

    cv2.putText(
        img,
        f"{1 / (time.perf_counter() - frame_timer):.2f} FPS",
        (25, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )

    if writer is not None:
        writer.write(img)

    cv2.imshow("VIDEO", img)


def main():

    display_video_feed(*parse_args())


if __name__ == "__main__":
    warnings.filterwarnings("ignore", message="torch.meshgrid")
    main()
