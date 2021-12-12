import torch
import torchmetrics
import numpy as np

CLASS_LABELS = {0: 'Without mask', 1: 'With mask', 2: 'With mask worn incorrectly'}
CLASS_COLORS = {0: 'tab:red', 1: 'tab:green', 2: 'tab:blue'}

def plot_pr_curve(model, dataloader, ax, iou_thres=0.5, per_class=False, **kwargs):
    """
    Add precision-recall curve @ iou_thres to the given matplotlib ax object.
    """

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model.eval()
    model.to(device)

    metric = torchmetrics.detection.map.MAP(compute_on_step=False, iou_thresholds=[iou_thres], max_detection_thresholds=[100])

    for img, labels in dataloader:

        img = img.to(device)

        prediction = model(img)

        for dict in prediction:
            for key in dict.keys():
                dict[key] = dict[key].detach().to("cpu")

        metric(prediction, labels)

    overall, map, _ = metric._calculate(metric._get_classes())

    recall = np.linspace(0, 1, 101)
    precision = overall['precision'][0, :, :, 0, 0] # size 101 x number of classes

    if per_class:
        for class_index in range(3):
            ax.plot(recall, precision[:, class_index], label=CLASS_LABELS[class_index], color=CLASS_COLORS[class_index])

    ax.plot(recall, torch.mean(precision, axis=1), **kwargs)