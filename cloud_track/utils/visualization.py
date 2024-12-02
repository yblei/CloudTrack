import os

import cv2
import numpy as np
import PIL
import torch


def save_image(image, name, basename="results"):
    path = os.path.join(basename, name)
    if isinstance(image, PIL.Image.Image):
        image.save(path)
    elif isinstance(image, torch.Tensor):
        image = image.squeeze().cpu().numpy()
        cv2.imwrite(path, image)
    elif isinstance(image, np.ndarray):
        cv2.imwrite(path, image)
    else:
        raise TypeError("image must be PIL.Image.Image or torch.Tensor")
    image.save(path)
