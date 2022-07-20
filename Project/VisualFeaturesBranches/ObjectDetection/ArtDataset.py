"""
Dataset class for Pytorch Models
"""
import random
from typing import List, Tuple

import PIL.Image
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from Project.AutoSimilarityCacheConfiguration.DataAccess import DataAccess


class ArtDataset(Dataset):
    """
    Dataset that provides various augmentations for object detection
    """
    def __init__(self, ids, width, height, for_training, p_vertical_flip, p_horizontal_flip,
                 brightness_jitter, contrast_jitter, saturation_jitter, hue_jitter, p_grayscale):
        super().__init__()
        assert width == height, 'Additional testing nescessary if width != height'
        self.__ids = ids
        self.__width = width
        self.__height = height
        self.__for_training = for_training
        self.__p_vertical_flip = p_vertical_flip
        self.__p_horizontal_flip = p_horizontal_flip
        self.__brightness_jitter = brightness_jitter
        self.__contrast_jitter = contrast_jitter
        self.__saturation_jitter = saturation_jitter
        self.__hue_jitter = hue_jitter
        self.__p_grayscale = p_grayscale
        self.__identifier_index_dict = None

    def __getitem__(self, index):
        da: DataAccess = DataAccess.instance
        identifier = self.__ids[index]

        boxes, labels = da.get_bounding_boxes_and_labels_for_identifier(identifier)

        pil = da.get_PIL_image_from_identifier(identifier, True)
        width_ratio = self.__width / pil.width
        height_ratio = self.__height / pil.height
        pil = pil.resize((self.__width, self.__height), PIL.Image.LANCZOS)
        img = np.array(pil) / 255
        new_boxes = []
        area = []

        def order_box_coordinates(box: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
            """
            Brings the box coordinates into the right order
            :param box: A box defined as a tuple of four coordinates
            :return: A box defined as a tuple of four sorted coordinates
            """
            x_1, y_1, x_2, y_2 = box
            if x_1 > x_2:
                x_1, x_2 = x_2, x_1
            if y_1 > y_2:
                y_1, y_2 = y_2, y_1
            return x_1, y_1, x_2, y_2

        for b in boxes:
            current_box = order_box_coordinates((int(b[0] * width_ratio), int(b[1] * height_ratio),
                                                 int(b[2] * width_ratio), int(b[3] * height_ratio)))
            new_boxes.append(current_box)
            area.append((current_box[2] - current_box[0]) * (current_box[3] - current_box[1]))

        boxes = new_boxes

        if self.__for_training:
            if random.random() > (1 - self.__p_vertical_flip):
                img = cv2.flip(img, 0)
                new_boxes: List[Tuple[int, int, int, int], ...] = []
                for b in boxes:
                    new_boxes.append(order_box_coordinates((b[0], pil.height - b[1], b[2], pil.height - b[3])))

                boxes = new_boxes
            if random.random() > (1 - self.__p_horizontal_flip):
                img = cv2.flip(img, 1)
                new_boxes: List[Tuple[int, int, int, int], ...] = []
                for b in boxes:
                    new_boxes.append(order_box_coordinates((pil.width - b[0], b[1], pil.width - b[2], b[3])))
                boxes = new_boxes

        img = torch.permute(torch.as_tensor(img, dtype=torch.float32), (2, 0, 1))
        if self.__for_training:
            img = transforms.Compose([
                transforms.ColorJitter(brightness=self.__brightness_jitter, contrast=self.__contrast_jitter,
                                       saturation=self.__saturation_jitter, hue=self.__hue_jitter),
                transforms.RandomGrayscale(self.__p_grayscale)
            ])(img)

        target = {
            'boxes': torch.as_tensor(boxes, dtype=torch.float32),
            'labels': torch.as_tensor(
                [da.get_class_label_for_bounding_box_label(label) for label in labels], dtype=torch.int64),
            'image_id': torch.tensor([index]),
            'iscrowd': torch.zeros((len(boxes),), dtype=torch.int64),
            'area': torch.as_tensor(area, dtype=torch.float32)
        }

        return img, target

    def get_item_by_identifier(self, identifier):
        if self.__identifier_index_dict is None:
            self.__identifier_index_dict = dict()
            for ind, i in enumerate(self.__ids):
                self.__identifier_index_dict[i] = ind
        return self.__getitem__(self.__identifier_index_dict[identifier])

    def __len__(self):
        return len(self.__ids)
