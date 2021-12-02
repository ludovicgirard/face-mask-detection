import torch
import torchmetrics


def evaluate_mAP(model, dataloader, **args):

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model.eval()
    model.to(device)

    metric = torchmetrics.detection.map.MAP(compute_on_step=False, **args)

    for img, labels in dataloader:

        img = img.to(device)

        prediction = model(img)

        # Can't store everything on GPU, so transfer to CPU
        for dict in prediction:
            for key in dict.keys():
                dict[key] = dict[key].detach().to("cpu")

        metric(prediction, labels)

    return metric.compute()
