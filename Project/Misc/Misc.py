"""
Misc.py
"""
import os

import numpy as np
from PIL import Image
from tqdm import tqdm

from Project.AutoSimilarityCacheConfiguration.DataAccess import DataAccess
from Project.Utils.Misc.OriginContainer import OriginContainer


def crop_border_of_all_sides(path: str) -> None:
    """
    Crops the background from an image and overwrites it. Warning: Repeated application can have unwanted effects.
    :param path: Path of the image that should be replaced
    :return: None
    """
    im = Image.open(path)
    # noinspection PyTypeChecker
    im_arr: np.array = np.array(im)
    left_col = im_arr[:, 0]
    top_row = im_arr[0, :]
    right_col = im_arr[:, im_arr.shape[1] - 1]
    bottom_row = im_arr[im_arr.shape[0] - 1, :]
    remove_from_top = 0
    remove_from_left = 0
    remove_from_bottom = 0
    remove_from_right = 0

    width: int = im_arr.shape[1]
    height: int = im_arr.shape[0]

    for i in range(1, width - 1):
        if (im_arr[:, i] == left_col).all():
            remove_from_left += 1
        else:
            break

    for i in range(width - 2, 0, -1):
        if (im_arr[:, i] == right_col).all():
            remove_from_right += 1
        else:
            break

    for i in range(1, height - 1):
        if (im_arr[i, :] == top_row).all():
            remove_from_top += 1
        else:
            break

    for i in range(height - 2, 0, -1):
        if (im_arr[i, :] == bottom_row).all():
            remove_from_bottom += 1
        else:
            break

    im_crop = im.crop((remove_from_left, remove_from_top, width - remove_from_right, height - remove_from_bottom))
    im_crop.save(path)


def open_image_and_print_tags_for_identifier(identifier: str, origin_container: OriginContainer) -> None:
    """
    :param identifier: Identifier of the artwork
    :param origin_container: OriginContainer specifying which tags to print
    :return: None
    """
    da: DataAccess = DataAccess.instance
    print(da.get_tag_tuples_from_identifier(identifier=identifier, origin_container=origin_container))
    da.get_PIL_image_from_identifier(identifier=identifier).show()


def save_pngs_as_jpgs(folder: str) -> None:
    """
    Every png image in the given folder gets compressed and saved as [file_name]_compressed.jpg
    :param folder: The folder in which images are located
    :return: None
    """
    assert os.path.exists(folder), f'Folder does not exist: {folder}'
    for _, _, files in os.walk(folder):
        for file in tqdm(files, desc='Compressing images'):
            if file.endswith('.png'):
                Image.open(os.path.join(folder, file)).convert('RGB').save(
                    os.path.join(folder, file[:len(file) - 4] + '_compressed.jpg'), optimize=True)
