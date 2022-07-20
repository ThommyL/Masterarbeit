"""
Interface.py
"""

from typing import Tuple, Dict, Optional, Callable, Union

import numpy as np

from Project.AutoSimilarityCache.Caching.SimilarityRankDatabase import SimilarityRankDatabase
from Project.AutoSimilarityCache.Caching.TagAverageSimilarityCache import TagAverageSimilarityCache
from Project.AutoSimilarityCache.Caching.TagVectorCache import TagVectorCache
from Project.AutoSimilarityCache.Misc import FirstStart, Misc
from Project.AutoSimilarityCache.TagOriginUtils import OriginProcessing
from Project.AutoSimilarityCache.TagOriginUtils.OriginProcessing import split_origin_container_to_origin_types
from Project.AutoSimilarityCacheConfiguration.ConfigurationMethods import get_similarity, get_type_equally_groups
from Project.Utils.Misc.OriginContainer import OriginContainer


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ FIRST START ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def first_start() -> None:
    """
    A method that should be executed on first start. First the config file is generated with standard parameters. Then
    the similarity databases are generated. Finally the config file is rewritten so that the first start parameter is
    then set to false.
    :return: None
    """
    FirstStart.first_start()


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ METHODS FOR FETCHING SIMILARITY RELATED INFORMATION ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def get_unweighted_similarity_between_tag_tuples(tag_origin_tuple_1: Tuple[any, str],
                                                 tag_origin_tuple_2: Tuple[any, str]) -> float:
    """
    :param tag_origin_tuple_1: The first tag tuple
    :param tag_origin_tuple_2: The second tag tuple
    :return: The similarity between the first and second tag tuple according to the methods in ConfigurationMethods.py
    """
    tvc: TagVectorCache = TagVectorCache.instance
    return get_similarity(
        vector_origin_tuple_1=(
            tvc.get_vector_for_tag_tuple(tag_origin_tuple=tag_origin_tuple_1), tag_origin_tuple_1[1]),
        vector_origin_tuple_2=(tvc.get_vector_for_tag_tuple(tag_origin_tuple=tag_origin_tuple_2), tag_origin_tuple_2[1])
    )


def __validate_weight_dict(weight_dict: Dict[str, float]):
    if weight_dict is None:
        return
    weight_sum = 0
    for k in weight_dict.keys():
        assert k in get_type_equally_groups()
        assert 0 <= weight_dict[k] <= 1
        weight_sum += weight_dict[k]
    assert weight_sum == 1


def __update_result_dict(result, current, weight):
    for k in current.keys():
        current[k] *= weight
    if result is None:
        return current
    else:
        for k in current.keys():
            result[k] += current[k]
    return result


def __apply_weighted_method(method: Callable, identifier: str, origin_container: OriginContainer, weight_dict
                            ) -> Dict[str, Union[int, float]]:
    __validate_weight_dict(weight_dict=weight_dict)
    type_origins_dict = split_origin_container_to_origin_types(origin_container)

    result: Optional[Dict[str, float]] = None
    for t in type_origins_dict.keys():
        current: Dict[str, float] = method(identifier=identifier, origin_container=type_origins_dict[t])
        result = __update_result_dict(result, current, weight_dict[t])
    return result


def get_similarities_for_id(identifier: str, origin_container: OriginContainer, weight_dict: Dict[str, float] = None
                            ) -> Dict[str, float]:
    """
    :param identifier: The entity for which to query
    :param origin_container: Which origins from tags should be considered for the search ('Des', 'Title', 'Exp', ...)
    :param weight_dict: None or Dictionary that contains type groups (see ConfigurationMethods.py) as keys and float
    values that sum up to 1 as weights
    :return: A tuple of Tuple[identifier, similarity]
    """
    srd: SimilarityRankDatabase = SimilarityRankDatabase.instance

    if weight_dict is None:
        return srd.get_similarities_for_id(
            identifier=identifier, origin_container=origin_container)
    return __apply_weighted_method(srd.get_similarities_for_id, identifier, origin_container, weight_dict)


def __apply_pairwise_weighted_method(
        method: Callable, id_1: str, id_2: str, origin_container: OriginContainer, weight_dict: Dict[str, float]
) -> Tuple[Dict[str, Union[int, float]], Dict[str, Union[int, float]]]:
    __validate_weight_dict(weight_dict=weight_dict)
    type_origins_dict = split_origin_container_to_origin_types(origin_container)

    result_1: Optional[Dict[str, float]] = None
    result_2: Optional[Dict[str, float]] = None

    for t in type_origins_dict.keys():
        current_1: Dict[str, float]
        current_2: Dict[str, float]
        current_1, current_2 = method(id_1=id_1, id_2=id_2, origin_container=type_origins_dict[t])
        result_1 = __update_result_dict(result_1, current_1, weight_dict[t])
        result_2 = __update_result_dict(result_2, current_2, weight_dict[t])
    return result_1, result_2


def get_similarities_for_pair(
        id_1: str, id_2: str, origin_container: OriginContainer, weight_dict: Dict[str, float] = None) -> Tuple[
        Dict[str, float], Dict[str, float]]:
    """
    :param id_1: The first identifier for which similarities should be returned
    :param id_2: The second identifier for which similarities should be returned
    :param origin_container: The origins of which tags should be considered
    :param weight_dict: None or Dictionary that contains type groups (see ConfigurationMethods.py) as keys and float
    values that sum up to 1 as weights
    :return: Tuple with 2 dictionaries corresponding to id_1 and id_2
    """
    srd: SimilarityRankDatabase = SimilarityRankDatabase.instance

    if weight_dict is None:
        return srd.get_similarities_for_pair(id_1=id_1, id_2=id_2, origin_container=origin_container)
    return __apply_pairwise_weighted_method(srd.get_similarities_for_pair, id_1, id_2, origin_container, weight_dict)


def get_ranks_for_pair(
        id_1: str, id_2: str, origin_container: OriginContainer, weight_dict: Dict[str, float] = None) -> Tuple[
        Dict[str, int], Dict[str, int]]:
    """
    :param id_1: The first identifier for which a rank dictionary should be returned
    :param id_2: The second identifier for which a rank dictionary should be returned
    :param origin_container: The origins of which tags should be considered
    :param weight_dict: None or Dictionary that contains type groups (see ConfigurationMethods.py) as keys and float
    values that sum up to 1 as weights
    :return: Tuple with 2 dictionaries corresponding to id_1 and id_2
    """
    srd: SimilarityRankDatabase = SimilarityRankDatabase.instance

    if weight_dict is None:
        return srd.get_ranks_for_pair(id_1=id_1, id_2=id_2, origin_container=origin_container)
    return __apply_pairwise_weighted_method(srd.get_ranks_for_pair, id_1, id_2, origin_container, weight_dict)


def get_ranks_for_id(
        identifier: str, origin_container: OriginContainer, weight_dict: Dict[str, float] = None) -> Dict[str, int]:
    """
    :param identifier: The sample for which to query
    :param origin_container: The origins of the tags that should be considered
    :param weight_dict: None or Dictionary that contains type groups (see ConfigurationMethods.py) as keys and float
    values that sum up to 1 as weights
    :return: A dictionary containing identifiers as keys and the respective ranks as value
    """
    srd: SimilarityRankDatabase = SimilarityRankDatabase.instance
    if weight_dict is None:
        return srd.get_ranks_for_id(identifier=identifier, origin_container=origin_container)
    return __apply_weighted_method(srd.get_ranks_for_id, identifier, origin_container, weight_dict)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ METHODS FOR CONVERSION OF ORIGIN INFORMATION ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def combined_origin_to_origin_tuple(combined_origin: str) -> Tuple[str, ...]:
    """
    :param combined_origin: A Combined Origin
    :return: Individual Origins of which the given combined origin consists of
    """
    return OriginProcessing.combined_origin_to_origin_tuple(combined_origin=combined_origin)


def generate_df_name(origin_container: OriginContainer, is_rank: bool) -> str:
    """
    :param origin_container: OriginContainer specifying which origins to consider for creating the name of the dataframe
    :param is_rank: True if name for a rank dataframe is to be returned, False if name of a similarity dataframe is to
    be returned
    :return: The name of the dataframe according to the specifications
    """
    return OriginProcessing.generate_df_name(origin_container=origin_container, is_rank=is_rank)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ METHODS FOR FETCHING AVERAGE SIMILARITY RELATED INFORMATION ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def get_average_similarity_of_tag_tuple(tag_tuple: Tuple[any, str], origin_container: OriginContainer) -> np.float32:
    """
    :param tag_tuple: The tag tuple for which to check the average similarity
    :param origin_container: OriginContainer specifying which origins to consider
    :return: The average similarity a tag has to all other tags
    """
    tasc: TagAverageSimilarityCache = TagAverageSimilarityCache.instance
    return tasc.get_average_similarity_of_tag_tuple(tag_tuple=tag_tuple, origin_container=origin_container)


def get_average_similarity_for_identifier(
        identifier: str, origin_container: OriginContainer, weight_dict: Dict[str, float] = None) -> float:
    """
    :param identifier: Identifier
    :param origin_container: OriginContainer specifying which origins to consider
    :param weight_dict: None or Dictionary that contains type groups (see ConfigurationMethods.py) as keys and float
    values that sum up to 1 as weights
    :return: Average similarity of the tags of the given identifier according to the input
    """
    tasc: TagAverageSimilarityCache = TagAverageSimilarityCache.instance
    if weight_dict is None:
        return tasc.get_average_similarity_for_identifier(
            identifier=identifier, combined_origin=generate_df_name(origin_container=origin_container, is_rank=False))
    __validate_weight_dict(weight_dict=weight_dict)
    type_origins_dict = split_origin_container_to_origin_types(origin_container)

    result: float = 0
    for t in type_origins_dict.keys():
        result += tasc.get_average_similarity_for_identifier(
            identifier=identifier, combined_origin=generate_df_name(
                origin_container=type_origins_dict[t], is_rank=False))
    return result


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ METHODS FOR FETCHING VECTORS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def get_vector_for_tag_tuple(tag_tuple: Tuple[any, str]) -> np.array:
    """
    :param tag_tuple: The tag for which to return the vector
    :return: The vector of the tag specified
    """
    tvc: TagVectorCache = TagVectorCache.instance
    return tvc.get_vector_for_tag_tuple(tag_origin_tuple=tag_tuple)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ METHODS FOR FILTERING TAG TUPLES ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def filter_and_optimize_data_tuples(identifier: str, origin_container: OriginContainer) -> Tuple[Tuple[any, str], ...]:
    """
    :param identifier: Identifier of which to return the filtered tags of.
    :param origin_container: OriginContainer specifying which origins should be considered
    :return: Filtered input
    """
    origin_type_dict = split_origin_container_to_origin_types(origin_container)

    result = []

    for k in origin_type_dict.keys():
        result += list(
            Misc.filter_and_optimize_data_tuples(identifier=identifier, origin_container=origin_type_dict[k]))
    return tuple(result)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ METHODS FOR USE IN TEST MODE ONLY ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def save_generated():
    """
    Saves the dataframes. Only to be used in test mode.
    :return: None
    """
    srd: SimilarityRankDatabase = SimilarityRankDatabase.instance
    return srd.save_generated()


def ensure_similarities_for_pair_are_generated(id_1: str, id_2: str, origin_container: OriginContainer) -> None:
    """
    Note: Only necessary in test mode
    This method checks whether entries for identifiers have been already generated and if not it does so in an
    efficient manner
    :param id_1: First identifier for which to check
    :param id_2: Second identifier for which to check
    :param origin_container: OriginContainer for which similarities should be checked / calculated
    :return: None
    """
    type_origins_dict = split_origin_container_to_origin_types(origin_container)
    srd: SimilarityRankDatabase = SimilarityRankDatabase.instance

    for t in type_origins_dict.keys():
        srd.ensure_similarities_for_pair_are_generated(id_1=id_1, id_2=id_2, origin_container=type_origins_dict[t])
