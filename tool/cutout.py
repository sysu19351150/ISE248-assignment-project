import numpy as np
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据裁剪，用以获取增强数据
class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, y_length,x_length):
        self.n_holes = n_holes
        self.x_length = x_length
        self.y_length=y_length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(2)
        w = img.size(3)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.y_length // 2, 0, h)
            y2 = np.clip(y + self.y_length // 2, 0, h)
            x1 = np.clip(x - self.x_length // 2, 0, w)
            x2 = np.clip(x + self.x_length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask).to(device)
        mask = mask.expand_as(img)
        img = img * mask

        return img