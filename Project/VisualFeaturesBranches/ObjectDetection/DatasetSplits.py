"""
Splits and saves Dataset
"""
import random
from typing import Optional

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.preprocessing import MultiLabelBinarizer

from Project.AutoSimilarityCacheConfiguration.DataAccess import DataAccess
from Project.VisualFeaturesBranches.ObjectDetection.ArtDataset import ArtDataset


class DatasetSplits:
    """
    Provides training-, test- and complete data set
    """

    def __init__(self, size, p_vertical_flip, p_horizontal_flip, brightness_jitter, contrast_jitter,
                 saturation_jitter, hue_jitter, p_grayscale, curriculum):
        self.__size = size
        self.__p_vertical_flip = p_vertical_flip
        self.__p_horizontal_flip = p_horizontal_flip
        self.__brightness_jitter = brightness_jitter
        self.__saturation_jitter = saturation_jitter
        self.__contrast_jitter = contrast_jitter
        self.__hue_jitter = hue_jitter
        self.__p_grayscale = p_grayscale
        self.__curriculum = curriculum

        da: DataAccess = DataAccess.instance
        lengths = [int(len(da.get_ids_for_which_bounding_boxes_exist()) * p) for p in (0.75, 0.25)]
        lengths[0] += len(da.get_ids_for_which_bounding_boxes_exist()) - sum(lengths)

        self.__test_set: ArtDataset

        identifiers = list(da.get_ids_for_which_bounding_boxes_exist())
        random.seed(0)
        random.shuffle(identifiers)

        X = []
        y = []
        for i in da.get_ids_for_which_bounding_boxes_exist():
            X.append(i)
            y.append([da.get_class_label_for_bounding_box_label(label) for label in
                      da.get_bounding_boxes_and_labels_for_identifier(i)[1]])

        mb: MultiLabelBinarizer = MultiLabelBinarizer(
            classes=[da.get_class_label_for_bounding_box_label(label) for label in
                     da.get_unique_labels_of_bounding_boxes()])
        mb.fit_transform(y)

        self.__training_ids = []
        self.__test_ids = []

        mskf = MultilabelStratifiedKFold(n_splits=3, random_state=0, shuffle=True)
        for train, test in mskf.split(X, mb.fit_transform(y)):
            current_train = []
            for t in train:
                current_train.append(X[t])
            self.__training_ids.append(current_train)

            current_test = []
            for t in test:
                current_test.append(X[t])
            self.__test_ids.append(current_test)

    def get_train(self, fold: int, epoch: Optional[int], max_epochs: Optional[int]) -> ArtDataset:
        """
        :param fold: The number of the fold [0 to 2]
        :param epoch: The current epoch, only relevant if curriculum=True
        :param max_epoch: The max epoch, only relevant if curriculum=True
        :return: Training Split
        """
        if self.__curriculum:
            factor = (epoch + 1) / max_epochs
        else:
            factor = 1
        return ArtDataset(ids=self.__training_ids[fold], width=self.__size, height=self.__size, for_training=True,
                          p_vertical_flip=self.__p_vertical_flip * factor,
                          p_horizontal_flip=self.__p_horizontal_flip * factor,
                          brightness_jitter=self.__brightness_jitter * factor,
                          contrast_jitter=self.__contrast_jitter * factor, saturation_jitter=self.__saturation_jitter,
                          hue_jitter=(self.__hue_jitter[0] * factor, self.__hue_jitter[1] * factor),
                          p_grayscale=self.__p_grayscale * factor)

    def get_test(self, fold) -> ArtDataset:
        """
        :param fold: The number of the fold [0 to 2]
        :return: Test Split
        """
        # noinspection PyArgumentEqualDefault
        return ArtDataset(
            ids=self.__test_ids[fold], width=self.__size, height=self.__size, for_training=False, p_vertical_flip=0,
            p_horizontal_flip=0, brightness_jitter=0, contrast_jitter=0, saturation_jitter=0, hue_jitter=(0, 0),
            p_grayscale=0)

    def get_training_all(self, epoch: int, multiplier_per_epoch: float) -> ArtDataset:
        """
        :return: Training Set that contains all identifiers of ground truth
        """
        da: DataAccess = DataAccess.instance
        if self.__curriculum:
            factor = multiplier_per_epoch * epoch
        else:
            factor = 1
        return ArtDataset(
            ids=da.get_ids_for_which_bounding_boxes_exist(), width=self.__size, height=self.__size, for_training=True,
            p_vertical_flip=self.__p_vertical_flip * factor, p_horizontal_flip=self.__p_horizontal_flip * factor,
            brightness_jitter=self.__brightness_jitter * factor, contrast_jitter=self.__contrast_jitter * factor,
            saturation_jitter=self.__saturation_jitter * factor, hue_jitter=self.__hue_jitter * factor,
            p_grayscale=self.__p_grayscale * factor)

    @property
    def number_of_splits(self) -> int:
        """
        :return: The number of splits
        """
        return len(self.__training_ids)
