"""
ConfigurationMethods
"""
import functools
import os.path
import pickle
import warnings
from typing import Tuple, Dict, Optional

import numpy as np

from Project.AutoSimilarityCacheConfiguration.DataAccess import DataAccess
from Project.Utils.IconclassCache import ICONCLASS_CACHE_GENERATION_WAS_SUCCESSFUL_PKL
from Project.Utils.Misc.Misc import cosine_distance
from Project.Utils.Misc.Nlp import NLP
from Project.Utils.Misc.OriginContainer import OriginContainer
from Project.Utils.TextTagProcessing.TagProcessing import filter_doubles

""" 
Origins and Rules
"""


def get_type_equally_groups(origin: str = None):
    """
    :param origin: None or origin container for which to query
    :return: If origin is none, then all possible groups are returned, else the group of which the origin is part of is
    returned.
    """
    if origin is None:
        return ('Semantic Type', 'Object Type')
    if origin in ('Des', 'Title', 'Exp', 'Icon', 'NotIcon', 'Obj'):
        return 'Semantic Type'
    else:
        raise Exception(f'Unrecognized Origin: {origin}')


def get_origin_type(origin: str) -> str:
    """
    :param origin: An origin
    :return: "Text" if the origin is associated with text labels
    """
    if origin in ('Des', 'Title', 'Exp', 'Icon', 'NotIcon'):
        return 'Text'
    if origin in ('Obj',):
        return 'Object'
    else:
        raise Exception('Unspecified Origin')


def __is_text_label(origin: str) -> bool:
    """
    :param origin: An origin
    :return: True if tags of this origin are text tags
    """
    return get_origin_type(origin=origin) == 'Text'


def __is_object_label(origin: str) -> bool:
    """
    :param origin: An origin
    :return: True if tags of this origin are text tags
    """
    return get_origin_type(origin=origin) == 'Object'


def get_similarity(vector_origin_tuple_1: Tuple[np.array, str], vector_origin_tuple_2: Tuple[np.array, str]) -> float:
    """
    Note: Needs to be threadsafe if multithreading is enabled
    :param vector_origin_tuple_1: First vector origin tuple
    :param vector_origin_tuple_2: Second vector origin tuple
    :return: The similarity between the given vectors considering their origins
    """
    if (__is_text_label(vector_origin_tuple_1[1]) or __is_object_label(vector_origin_tuple_1[1])) and \
            (__is_text_label(vector_origin_tuple_2[1]) or __is_object_label(vector_origin_tuple_2[1])):
        return cosine_distance(vector_origin_tuple_1[0], vector_origin_tuple_2[0])
    raise Exception(f'Could not compute similarity between tag_origin_tuples of origin '
                    f'{vector_origin_tuple_1[1]} and {vector_origin_tuple_2[1]}')


@functools.lru_cache()
def get_similarity_weight(identifier: str, tag_origin_tuple: Tuple[np.array, str]) -> float:
    """
    Note: Needs to be threadsafe if multithreading is enabled
    :param identifier: The identifier to which the tag origin tuple belongs to
    :param tag_origin_tuple: The tag origin tuple for which to fetch the weight
    :return: The weight according to the input
    """
    if __is_text_label(tag_origin_tuple[1]) or __is_object_label(tag_origin_tuple[1]):
        da: DataAccess = DataAccess.instance
        return da.get_weight_for_identifier_tag_tuple(identifier=identifier, tag_origin_tuple=tag_origin_tuple)
    else:
        raise Exception('Unhandled case')


def get_vector(tag_origin_tuple: Tuple[any, str]) -> np.ndarray:
    """
    Note: Does not need to be thread safe
    :param tag_origin_tuple: Tuple containing a tag as well as an origin
    :return: Vector as numpy array
    """
    if __is_text_label(tag_origin_tuple[1]) or __is_object_label(tag_origin_tuple[1]):
        nlp: NLP = NLP.instance
        return np.array(nlp.nlp(tag_origin_tuple[0]).vector)


def get_possible_origins() -> Tuple[str, ...]:
    """
    Note: The combined origins must contain the origin in their names in the order that is given in this method
    :return: The possible origins
    """
    return 'Des', 'Title', 'Exp', 'Icon', 'NotIcon', 'Obj'


def get_possible_combined_origins(origin_tuple: Optional[Tuple[str, ...]]) -> Tuple[str, ...]:
    """
    :param origin_tuple: Origins that are active
    :return: Combinations of origins that are allowed given active origins, if origin_tuple is None, return all unique
    combinations
    """
    if origin_tuple is None:
        return 'Exp', 'Title', 'Title&Exp', 'Des', 'Des&Exp', 'Des&Title', 'Des&Title&Exp', 'NotIcon', \
               'Title&NotIcon', 'Icon', 'Obj', 'Exp&Obj', 'Title&Obj', 'Title&Exp&Obj', 'NotIcon&Obj', \
               'Title&NotIcon&Obj'
    if 'Icon' in origin_tuple:
        return 'Icon',
    if 'NotIcon' in origin_tuple:
        return 'Obj', 'NotIcon', 'Title', 'Title&Obj', 'Title&NotIcon', 'NotIcon&Obj', 'Title&NotIcon&Obj'
    return 'Exp', 'Title', 'Title&Exp', 'Des', 'Des&Exp', 'Des&Title', 'Des&Title&Exp', 'Obj', 'Exp&Obj', 'Title&Obj', \
           'Title&Exp&Obj'


def get_generally_possible_combined_origins() -> Tuple[str, ...]:
    """
    :return: Combinations of origins that are possible
    """
    return 'Icon', 'NotIcon', 'Exp', 'Title', 'Title&NotIcon', 'Title&Exp', 'Des', 'Des&Exp', 'Des&Title', \
           'Des&Title&Exp', 'Obj', "Title&Obj", "Exp&Obj", "Title&Exp&Obj", 'NotIcon&Obj', 'Title&NotIcon&Obj'


def validate_configuration(origin_tuple: Tuple[str, ...], active_combined_origins: Tuple[str, ...]) -> None:
    """
    Note: Does not need to be thread safe
    :param origin_tuple: Tuple specifying which origins are enabled
    :param active_combined_origins: combined origins which are enabled
    :raises InvalidConfigurationException: If configuration is not valid, this exception is raised
    :return: None
    """
    des: bool = 'Des' in origin_tuple
    icon: bool = 'Icon' in origin_tuple

    if des:
        warnings.warn('Des Tags should not be used, since it makes the calculations very inefficient')

    if icon:
        if os.path.exists(ICONCLASS_CACHE_GENERATION_WAS_SUCCESSFUL_PKL + '.pkl'):
            with open(ICONCLASS_CACHE_GENERATION_WAS_SUCCESSFUL_PKL + '.pkl', 'rb') as f:
                if not pickle.load(f):
                    warnings.warn('Iconclass cache was not yet successfully initialized. If you are currently '
                                  'generating this cache, ignore this warning now and check if it reappears at '
                                  'the next start. If so, the iconclass measurement is not accurate.')

    allowed: Tuple[str, ...]

    for co in active_combined_origins:
        assert co in get_possible_combined_origins(origin_tuple)


def filter_in_advance(identifier, origin_container: OriginContainer) -> Tuple[Tuple[any, str], ...]:
    """
    Note: Needs to be threadsafe if multithreading is enabled
    :param identifier: identifier for which to return the filtered results
    :param origin_container: origins which to consider
    :return: A filtered result according to the input.
    """
    da: DataAccess = DataAccess.instance
    tags = da.get_tag_tuples_from_identifier(identifier=identifier, origin_container=origin_container)
    if not tags:
        return tags
    origin_group = None
    for o in tags:
        if origin_group is None:
            origin_group = get_type_equally_groups(o[1])
        elif origin_group != get_type_equally_groups(o[1]):
            raise Exception('Origin Mismatch')

    if origin_group == 'Semantic Type':
        return filter_doubles(identifier=identifier, tag_tuples=tuple(tags))


def reduce_to_one_of_type(
        tuples: Tuple[Tuple[any, str], ...]) -> Tuple[Tuple[Tuple[str, any], ...], Dict[Tuple[any, str],
                                                                                        Tuple[any, str]]]:
    """
    Note: Needs to be threadsafe if multithreading is enabled
    This is not only a measure for increasing efficiency, but also a measure to avoid bias.
    Efficiency: Say the tuples ('Text', 'SomeTextType1') and ('Text', 'SomeTextType2') yield the same vector. Then it
    would be wastefull to calculate it twice. Therefore the same_as_dict that is returned as second value, references
    the tuples that do not have to be generated. In the example above the return value of this method would be:
    ((('Text', 'SomeTextType1'),), {('Text', 'SomeTextType2'): ('Text', 'SomeTextType1')}).
    Reducing Bias: The AverageSimilarityCache compares all tuples against each other. Say that there are 5 tuples:
    ('Text', 'SomeTextType1'), ('Text', 'SomeTextType2'), ('Text', 'SomeTextType3'), ('Milk', 'SomeTextType1'),
    ('Sugar', 'SomeTextType2'). If we compare these against each other, then the result would be that 'Text' would have
    a high average similarity. However 'Text' should only be considered once. If this is done then the result will be
    the opposite and 'Text' would have a low average similarity.
    :param tuples: Tuples which to reduce
    :return: Tuple with reduced input at index 0 and a dictionary containing the tuples that were filtered out as keys
    and the tuples that yield the same result and were not reduced as values (in other words, the tuples from which
    results can be copied)
    """
    result: Dict[Tuple[str, any], Tuple[str, any]] = dict()
    same_as_dict: Dict[Tuple[any, str], Tuple[any, str]] = dict()
    for t in tuples:
        current: Tuple[any, str] = (t[0], get_origin_type(t[1]))

        if current in result.keys():
            reference: Tuple[any, str] = result[current]
            if reference != t:
                same_as_dict[t] = reference
        else:
            result[current] = t

    return tuple(result.values()), same_as_dict


def check_additional_constraints() -> None:
    """
    This method is called by the "check_constraints" method of the AutoSimilarityCache. it can be used to check
    constraints.
    :return: None
    """
    pass
