import torch
import torchmetrics
import numpy as np

def plot_pr_curve(model, dataloader, ax, label, iou_thres=0.5):
    """
    Add precision-recall curve @ iou_thres to the given matplotlib ax object, with given label for the legend.
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

    print(map)

    recall = np.linspace(0, 1, 101)
    precision = torch.mean(overall['precision'][0, :, :, 0, 0], axis=1)

    ax.plot(recall, precision, label=label)
    