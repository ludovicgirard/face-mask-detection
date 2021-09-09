import torch
import torch.nn.functional as F


class RandomCrop:

    def __init__(self, shape):

        self.shape = shape

    def __call__(self, img, labels):

        # Pad if needed
        padx, pady = 0, 0
        needs_padding = False
        if img.shape[-2] < self.shape[0]:
            padx = int((self.shape[0] - img.shape[-2]) // 2) + 1
            needs_padding = True
        if img.shape[-1] < self.shape[1]:
            pady = int((self.shape[1] - img.shape[-1]) // 2) + 1
            needs_padding = True
        if needs_padding:
            img = F.pad(img, (pady, pady, padx, padx), 'constant', 0)
            labels = F.pad(labels, (pady, pady, padx, padx), 'constant', 0)

        # Crop

        x_start = torch.randint(
            0, high=img.shape[-2] - self.shape[0] + 1, size=(1,))
        y_start = torch.randint(
            0, high=img.shape[-1] - self.shape[1] + 1, size=(1,))

        img = img[..., x_start:x_start + self.shape[0],
                  y_start:y_start + self.shape[1]]
        labels = labels[..., x_start:x_start + self.shape[0],
                        y_start:y_start + self.shape[1]]

        return img, labels


if __name__ == '__main__':

    img = torch.rand((2, 3, 123, 126))
    labels = torch.rand((2, 3, 123, 126))

    pad = RandomCrop((256, 256))

    img, labels = pad(img, labels)
    print(img.shape)
    print(labels.shape)
