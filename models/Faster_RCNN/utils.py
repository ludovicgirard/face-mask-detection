import torch


def precompute_mIOU(img, pred, labels, n_classes=4, threshold=0.5):

    # Unpack both pred and labels in dense tensors
    miou = torch.zeros((n_classes, 2))

    for idx in range(len(pred)):
        pred_dense = torch.zeros((n_classes, img.shape[-2], img.shape[-1]))
        label_dense = torch.zeros((n_classes, img.shape[-2], img.shape[-1]))

        pred_ = pred[idx]
        labels_ = labels[idx]

        for idx2 in range(len(pred_["labels"])):

            label = pred_["labels"][idx2]
            boxes = pred_["boxes"][idx2]

            pred_dense[
                label, int(boxes[0]) : int(boxes[2]), int(boxes[1]) : int(boxes[3])
            ] = 1.0

        for idx2 in range(len(labels_["labels"])):

            label = labels_["labels"][idx2]
            boxes = labels_["boxes"][idx2]

            label_dense[
                label, int(boxes[0]) : int(boxes[2]), int(boxes[1]) : int(boxes[3])
            ] = 1.0

        for c in range(1, n_classes):

            miou[c, 0] = torch.logical_and(
                pred_dense[c, :, :] == 1, label_dense[c, :, :] == 1
            ).sum()
            miou[c, 1] = torch.logical_or(
                pred_dense[c, :, :] == 1, label_dense[c, :, :] == 1
            ).sum()

    return miou
