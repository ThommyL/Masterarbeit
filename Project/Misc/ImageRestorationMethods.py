"""
Methods for optimizing images before they are used as input for a neural network.
These methods are adapted from one of my previous works.
"""

import copy

import PIL
import cv2
import numpy as np
from PIL import ImageOps
from skimage import img_as_float
from skimage.restoration import denoise_wavelet


def __denoise_image(img: np.array) -> np.array:
    img = img_as_float(copy.deepcopy(img).astype(np.uint8))
    return np.clip(np.multiply(denoise_wavelet(img, multichannel=True, convert2ycbcr=True), 255),
                   a_min=0, a_max=255)


def __sharpen_image(img: np.array) -> np.array:
    img = copy.deepcopy(img)
    return np.clip(np.subtract(np.asarray(img, dtype=np.float64) * 2, __smoothen_image(img)), a_min=0, a_max=255)


def __apply_ideal_contrast(img: np.array) -> np.array:
    # noinspection PyTypeChecker
    return np.asarray(ImageOps.autocontrast(PIL.Image.fromarray(np.uint8(copy.deepcopy(img))), cutoff=2, ignore=2))


def __smoothen_image(img: np.ndarray):
    mul = 1 / 9
    kernel = np.array([[mul, mul, mul], [mul, mul, mul], [mul, mul, mul]])
    return cv2.filter2D(np.asarray(np.array(copy.deepcopy(img)), dtype=np.float64), -1, kernel)


def restore_image(pil_image) -> PIL.Image:
    """
    :param pil_image: PIL Image
    :return: PIL Image that is denoised and sharpened with an ideal contrast
    """
    return PIL.Image.fromarray(__apply_ideal_contrast(__sharpen_image(__denoise_image(np.array(pil_image)))))
