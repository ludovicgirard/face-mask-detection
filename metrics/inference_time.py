import time

import torch


def evaluate_inference_time(model, dataloader):

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model.eval()
    model.to(device)
    times = []

    for img, labels in dataloader:

        img = img.to(device)

        timer = time.perf_counter()
        prediction = model(img)
        times.append(time.perf_counter() - timer)

    times = torch.tensor(times)

    result = {
        "Average inference time": torch.mean(times),
        "Standard deviation": torch.std(times),
        "Minimum inference time": torch.min(times),
        "Maximum inference time": torch.max(times),
        "Median inference time": torch.median(times),
    }

    return result
