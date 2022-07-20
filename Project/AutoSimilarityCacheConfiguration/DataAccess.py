"""
DataAccess
"""

import functools
import os
import pickle
from difflib import SequenceMatcher
from typing import Tuple, Optional, Dict, Set, List

import matplotlib.image
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision.io import read_image
from tqdm import tqdm

from Project.AutoSimilarityCacheConfiguration import WEIGHT_CACHE_PATH, CREATORS_CACHE_PATH, TITLES_CACHE_PATH, \
    ALL_LABELS_CACHE_PATH, ENHANCED_IMAGES_PATH, BOUNDING_BOXES_PKL_PATH, \
    TREE_ANNOTATIONS_PKL_PATH, PERSON_ANNOTATIONS_PKL_PATH
from Project.Misc.ImageRestorationMethods import restore_image
from Project.Utils.Misc.OriginContainer import OriginContainer
from Project.Utils.Misc.Singleton import Singleton


@Singleton
class DataAccess:
    """
    Singleton Pattern Class that loads the dataset and provides useful functions to simplify access to the data.
    Some methods (see method docs) are required by the AutoSimilarityCache.
    """
    def __init__(self):
        self.__df: pd.DataFrame = pd.read_pickle(os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 'Notebooks', 'Dataset_Tasks',
            'cleaned_dataframe.pkl'))
        self.__weight_cache: Optional[pd.DataFrame] = None

        self.__weight_cache_path = WEIGHT_CACHE_PATH + '.pkl'
        self.__creators_cache_path = CREATORS_CACHE_PATH + '.pkl'
        self.__titles_cache_path = TITLES_CACHE_PATH + '.pkl'
        self.__all_labels_cache = ALL_LABELS_CACHE_PATH + '.pkl'
        self.__creators: Optional[Set[str]] = None
        self.__titles: Optional[Set[str]] = None

        if not os.path.exists(ENHANCED_IMAGES_PATH):
            os.makedirs(ENHANCED_IMAGES_PATH)
            for i in tqdm(self.get_ids(), desc=f'Enhancing Images'):
                cur_path = os.path.join(ENHANCED_IMAGES_PATH, i + '.png')
                restore_image(self.get_PIL_image_from_identifier(i)).save(cur_path)

        self.__boxes_dict: Optional[Dict[str, Tuple[Tuple[int, int, int, int], ...], Tuple[str, ...]]] = None
        if os.path.exists(BOUNDING_BOXES_PKL_PATH):
            with open(BOUNDING_BOXES_PKL_PATH, 'rb') as f:
                self.__boxes_dict = pickle.load(f)

        self.__trees_dict: Optional[Dict[str, bool]] = None
        if os.path.exists(TREE_ANNOTATIONS_PKL_PATH):
            with open(TREE_ANNOTATIONS_PKL_PATH, 'rb') as f:
                self.__trees_dict = pickle.load(f)

        self.__persons_dict: Optional[Dict[str, bool]] = None
        if os.path.exists(PERSON_ANNOTATIONS_PKL_PATH):
            with open(PERSON_ANNOTATIONS_PKL_PATH, 'rb') as f:
                self.__persons_dict = pickle.load(f)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ REQUIRED ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @functools.lru_cache()
    def get_ids(self) -> Tuple[str, ...]:
        """
        Note: REQUIRED
        :return: A tuple of all identifiers of the cleaned dataset
        """
        return tuple(self.__df['Identifier'])

    @functools.lru_cache()
    def get_tag_tuples_from_identifier(
            self, identifier: str, origin_container: OriginContainer) -> Tuple[Tuple[any, str], ...]:
        """
        Note: REQUIRED
        :param origin_container: OriginContainer specifying the origins from which tuples should be returned
        :param identifier: The string that identifies the sample in the dataframe
        :return: A tuple of tuples containing the tags and their origins
        """
        self.is_valid_identifier(identifier)
        return tuple((tt[0], tt[1]) for tt in self.__df.loc[
            self.__df['Identifier'] == identifier, 'GeneratedTags'].iloc[0] if origin_container.origin_is_true(tt[1]))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ TRIVIAL ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @functools.lru_cache()
    def is_valid_identifier(self, identifier: str) -> bool:
        """
        Note: REQUIRED
        :param identifier: identifier for which to check whether it is in the cleaned dataframe
        :return: True if identifier is in the cleaned dataframe, False otherwise
        """
        assert isinstance(identifier, str), 'identifier must be of type str'
        return identifier in self.get_ids()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ADDITION ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __get_weight_cache(self):
        if self.__weight_cache is not None:
            return self.__weight_cache

        from Project.Utils.ConfigurationUtils.ConfigLoader import ConfigLoader
        cl: ConfigLoader = ConfigLoader.instance

        # Since the weights are accessed quite frequently, a cache is generated for them
        if not cl.data_cleaning_in_progress:
            self.__weight_cache: pd.DataFrame

            if not os.path.exists(self.__weight_cache_path):
                from Project.Utils.Misc.ProcessAndThreadHandlers import ProcessHandler
                ph: ProcessHandler = ProcessHandler.instance
                while True:
                    if not ph.acquire_data_generation_lock():
                        # If this was generated in the meantime
                        if os.path.exists(self.__weight_cache_path):
                            self.__weight_cache = pd.read_pickle(self.__weight_cache_path)
                            break

                        all_rows: Dict[Tuple[str, any, str], float] = dict()
                        for i in tqdm(self.get_ids(), 'Initializing DataAccess Cache'):
                            for tt in self.__df.loc[self.__df['Identifier'] == i, 'GeneratedTags'].iloc[0]:
                                cur_tag_tuple: Tuple[str, any, str] = (i, tt[0], tt[1])
                                assert cl.data_cleaning_in_progress or (
                                        cur_tag_tuple not in all_rows.keys() or all_rows[cur_tag_tuple] == tt[2]), \
                                    f'The tag tuple {tt} is found multiple times with different weights associated ' \
                                    f'to it.'
                                all_rows[cur_tag_tuple] = tt[2]
                        weight_cache: pd.DataFrame = pd.DataFrame(
                            all_rows.items(), columns=('IdentifierTagTuple', 'Weight'))
                        weight_cache.to_pickle(self.__weight_cache_path)
                        self.__weight_cache = weight_cache
                        break
                ph.release_data_generation_lock()
            else:
                self.__weight_cache = pd.read_pickle(self.__weight_cache_path)

        return self.__weight_cache

    @functools.lru_cache()
    def get_weight_for_identifier_tag_tuple(self, identifier: str, tag_origin_tuple: Tuple[any, str]) -> float:
        """
        :param identifier: The identifier to which the tag tuple belongs to
        :param tag_origin_tuple: The tag tuple for which the associated weight should be returned
        :return: The weight associated to the tag tuple
        """
        weight_cache = self.__get_weight_cache()
        return weight_cache.loc[weight_cache['IdentifierTagTuple'] ==
                                (identifier, tag_origin_tuple[0], tag_origin_tuple[1]), 'Weight'].iloc[0]

    def get_title_for_identifier(self, identifier: str) -> str:
        """
        :param identifier: The string that identifies the sample in the dataframe
        :return: The name of the artwork associated with the given identifier
        """
        self.is_valid_identifier(identifier)
        return self.__df.loc[self.__df['Identifier'] == identifier, 'Title'].iloc[0]

    def get_creator_for_identifier(self, identifier: str) -> str:
        """
        :param identifier: The string that identifies the sample in the dataframe
        :return: The name of the artist that created the artwork associated with the given identifier
        """
        self.is_valid_identifier(identifier)
        artist = self.__df.loc[self.__df['Identifier'] == identifier, 'Creator'].iloc[0]
        if pd.isna(artist):
            return 'Unknown Artist'
        return artist

    def get_iconclass_tags_from_identifier(self, identifier: str) -> Tuple[str, ...]:
        """
        :param identifier: The string that identifies the sample in the dataframe
        :return: A tuple of the unedited iconclass tags of the identifier
        """
        self.is_valid_identifier(identifier)
        return self.__df.loc[self.__df['Identifier'] == identifier, 'IconclassTags'].iloc[0]

    def get_expert_tags_from_identifier(self, identifier: str) -> Tuple[str, ...]:
        """
        :param identifier: The string that identifies the sample in the dataframe
        :return: A tuple of the unedited expert tags of the identifier
        """
        self.is_valid_identifier(identifier)
        return self.__df.loc[self.__df['Identifier'] == identifier, 'ExpertTags'].iloc[0]

    @functools.lru_cache()
    def __get_iconclass_texts(self):
        from Project.Utils.IconclassCache import ICONCLASS_CATEGORY_TO_NAME_PKL
        df = pd.read_pickle(ICONCLASS_CATEGORY_TO_NAME_PKL + '.pkl')
        all_texts: Set[str, ...] = set()

        for _, row in tqdm(df.iterrows(), desc="Collecting all iconclass tags"):
            all_texts.add(row['Text'].iloc[0])

        return all_texts

    def get_object_class_from_identifier(self, identifier: str) -> str:
        """
        :param identifier: The string that identifies the sample in the dataframe
        :return: The object class associated with the identifier
        """
        self.is_valid_identifier(identifier)
        result = self.__df.loc[self.__df['Identifier'] == identifier, 'ObjectClass'].iloc[0]
        if pd.isna(result):
            return ''
        return result

    def get_material_technique_from_identifier(self, identifier: str) -> str:
        """
        :param identifier: The string that identifies the sample in the dataframe
        :return: The material technique associated with the identifier
        """
        self.is_valid_identifier(identifier)
        result = self.__df.loc[self.__df['Identifier'] == identifier, 'MaterialTechnique'].iloc[0]
        if pd.isna(result):
            return ''
        return result

    def get_temporal_from_identifier(self, identifier: str) -> str:
        """
        :param identifier: The string that identifies the sample in the dataframe
        :return: The Epoch associated with the identifier
        """
        self.is_valid_identifier(identifier)
        result = self.__df.loc[self.__df['Identifier'] == identifier, 'Temporal'].iloc[0]
        if pd.isna(result):
            return ''
        return result

    def get_year_estimate_from_identifier(self, identifier: str) -> int:
        """
        :param identifier: The string that identifies the sample in the dataframe
        :return: The year estimate associated with the identifier
        """
        self.is_valid_identifier(identifier)
        return self.__df.loc[self.__df['Identifier'] == identifier, 'YearEstimate'].iloc[0]

    def get_creation_date_from_identifier(self, identifier: str) -> str:
        """
        :param identifier: The string that identifies the sample in the dataframe
        :return: The creation date value associated with the identifier
        """
        self.is_valid_identifier(identifier)
        result = self.__df.loc[self.__df['Identifier'] == identifier, 'CreationDate'].iloc[0]
        if pd.isna(result):
            return ''
        return result

    def get_description_from_identifier(self, identifier: str) -> str:
        """
        :param identifier: The string that identifies the sample in the dataframe
        :return: The description associated with the identifier
        """
        self.is_valid_identifier(identifier)
        result = self.__df.loc[self.__df['Identifier'] == identifier, 'Description'].iloc[0]
        if pd.isna(result):
            return ''
        return result

    @staticmethod
    def get_image_path(identifier: str, enhanced=False) -> str:
        """
        :param identifier: Identifier for which to get the image path
        :param enhanced: If True, the path of the enhanced image is returned instead
        :return: The path to the image as requested
        """
        if enhanced:
            return os.path.abspath(os.path.join(ENHANCED_IMAGES_PATH, identifier + '.png'))
        return os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Notebooks', 'Dataset_Tasks',
                                            'images', identifier + '.png'))

    def get_torchvision_image_from_identifier(self, identifier, enhanced=False) -> torch.Tensor:
        """
        :param identifier: Identifier for which to get the image path
        :param enhanced: If True, the path of the enhanced image is returned instead
        :return: The requested image as torch.Tensor
        """
        return read_image(self.get_image_path(identifier, enhanced))

    # noinspection PyPep8Naming
    def get_PIL_image_from_identifier(self, identifier: str, enhanced=False) -> Image:
        """
        :param identifier: Identifier for which to get the image path
        :param enhanced: If True, the path of the enhanced image is returned instead
        :return: The image associated to the given identifier as PIL Image
        """
        # noinspection PyTypeChecker
        return Image.open(self.get_image_path(identifier=identifier, enhanced=enhanced)).convert('RGB')

    def get_matplotlib_image_from_identifier(self, identifier: str, enhanced=False) -> np.ndarray:
        """
        :param identifier: Identifier for which to get the image path
        :param enhanced: If True, the path of the enhanced image is returned instead
        :return: matplotlib.image (numpy array)
        """
        return matplotlib.image.imread(self.get_image_path(identifier=identifier, enhanced=enhanced), format='png')

    def get_image_as_np_array_from_identifier(self, identifier: str, enhanced=False) -> np.ndarray:
        """
        :param identifier: Identifier for which to get the image path
        :param enhanced: If True, the path of the enhanced image is returned instead
        :return: Image as numpy array
        """
        return np.array(self.get_matplotlib_image_from_identifier(identifier, enhanced) * 256, dtype=np.uint8)

    def get_all_creators(self) -> Set[str]:
        """
        :return: A set of all creators in the dataset
        """
        if self.__creators is None:
            if os.path.exists(self.__creators_cache_path):
                with open(self.__creators_cache_path, 'rb') as f:
                    self.__creators = pickle.load(f)
            else:
                creators: Set[str] = set()
                for i in tqdm(self.get_ids(), desc='Collecting all unique creators in the dataset'):
                    current_creator: str = self.get_creator_for_identifier(identifier=i)

                    if not pd.isna(current_creator):
                        creators.add(current_creator)
                with open(self.__creators_cache_path, 'wb+') as f:
                    pickle.dump(creators, f)
                self.__creators = creators
        return self.__creators

    def get_all_titles(self) -> Set[str]:
        """
        :return: A set of all titles in the dataset
        """
        if self.__titles is None:
            if os.path.exists(self.__titles_cache_path):
                with open(self.__titles_cache_path, 'rb') as f:
                    self.__titles = pickle.load(f)
            else:
                titles: Set[str] = set()
                for i in tqdm(self.get_ids(), desc='Collecting all unique artists in the dataset'):
                    current_titles: str = self.get_title_for_identifier(identifier=i)
                    titles.add(current_titles)
                with open(self.__titles_cache_path, 'wb+') as f:
                    pickle.dump(titles, f)
                self.__titles = titles
        return self.__titles

    def get_closest_creators(self, creator: str):
        """
        :param creator: The approximate name of the creator
        :return: All creators of the dataset ordered by their distance to the input
        """
        result: Dict[str, float] = dict()

        for c in self.get_all_creators():
            result[c] = SequenceMatcher(a=creator, b=c, autojunk=False).ratio()
        return tuple(t[0] for t in sorted(result.items(), key=lambda x: x[1], reverse=True))

    def get_closest_titles(self, title: str):
        """
        :param title: The approximate name of the artwork
        :return: All the artworks in the dataset, ordered by their distance to the input
        """
        result: Dict[str, float] = dict()

        for c in self.get_all_titles():
            result[c] = SequenceMatcher(a=title, b=c, autojunk=False).ratio()
        return tuple(t[0] for t in sorted(result.items(), key=lambda x: x[1], reverse=True))

    @functools.lru_cache()
    def get_all_title_and_exp_labels(self):
        """
        :return: All unique labels of origins 'Title' or 'Exp' in a set order
        """
        if not os.path.exists(self.__all_labels_cache):
            from Project.AutoSimilarityCache.Interface import get_vector_for_tag_tuple

            all_labels: List[Tuple[any, str]] = []
            for i in tqdm(self.get_ids(), desc='Collecting labels'):
                for tt in self.get_tag_tuples_from_identifier(i, OriginContainer(('Title', 'Exp'))):
                    all_labels.append(get_vector_for_tag_tuple(tt))
            with open(self.__all_labels_cache, 'wb+') as f:
                pickle.dump(all_labels, f)
        with open(self.__all_labels_cache, 'rb') as f:
            return pickle.load(f)

    @functools.lru_cache()
    def get_ids_for_which_bounding_boxes_exist(self) -> Tuple[str]:
        """
        :return: Return those ids for which bounding boxes were drawn
        """
        return tuple(self.__boxes_dict.keys())

    def get_bounding_boxes_and_labels_for_identifier(
            self, identifier: str) -> Tuple[Tuple[Tuple[int, int, int, int], ...], Tuple[str, ...]]:
        """
        :param identifier: Image identifier
        :return: Tuple containing Tuple of bounding boxes and Tuple of labels
        """
        return self.__boxes_dict[identifier] if identifier in self.__boxes_dict.keys() else (tuple(), tuple())

    @functools.lru_cache()
    def get_unique_labels_of_bounding_boxes(self) -> Tuple[str]:
        """
        :return: Get the set of all existing labels of bounding boxes as an ordered Tuple
        """
        all_labels: Set[str] = set()
        for k in self.__boxes_dict.keys():
            for label in self.__boxes_dict[k][1]:
                all_labels.add(label)
        return tuple(sorted(all_labels))

    def get_class_label_for_bounding_box_label(self, label) -> int:
        """
        :param label: A label of a bounding box
        :return: An integer which represents the label
        """
        unique_labels: Tuple[str] = self.get_unique_labels_of_bounding_boxes()
        return unique_labels.index(label) + 1

    def get_class_label_for_index(self, index: int):
        """
        :param index: An integer representing a label
        :return: A bounding box label
        """
        unique_labels: Tuple[str] = self.get_unique_labels_of_bounding_boxes()
        return unique_labels[index - 1]

    def get_ids_for_which_tree_annotation_exists(self):
        """
        :return: All identifiers which have been annotated with whether the artworks contain a tree or not
        """
        return tuple(self.__trees_dict.keys())

    def get_tree_annotation_for_identifier(self, identifier: str):
        """
        :param identifier: An identifier of an artwork
        :return: True if artwork contains at least one tree, False otherwise
        """
        return self.__trees_dict[identifier]

    def get_ids_for_which_person_annotation_exists(self):
        """
        :return: All identifiers which have been annotated with whether the artworks contain a person or not
        """
        return tuple(self.__persons_dict.keys())

    def get_person_annotation_for_identifier(self, identifier: str):
        """
        :param identifier: An identifier of an artwork
        :return: True if artwork contains at least one person, False otherwise
        """
        return self.__persons_dict[identifier]
