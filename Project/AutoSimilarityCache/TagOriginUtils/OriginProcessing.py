"""
OriginProcessing
"""

import functools
import os
import pickle
import shutil
from typing import Tuple, List, Set, Dict

from Project.AutoSimilarityCache.Caching import ALL_VALUES_CHECKED, TAG_VECTOR_CACHE_PATH
from Project.AutoSimilarityCacheConfiguration.ConfigurationMethods import get_type_equally_groups
from Project.Utils.ConfigurationUtils.ConfigLoader import ConfigLoader
from Project.Utils.FilterCache import FILTER_CACHES_PATH
from Project.Utils.Misc.OriginContainer import OriginContainer

"""
All things that concerns the processing of tags, should be done in this file. 
"""


def logical_or_for_origin_containers(
        origin_container_1: OriginContainer, origin_container_2: OriginContainer) -> OriginContainer:
    """
    :param origin_container_1: First OriginContainer
    :param origin_container_2: Second OriginContainer
    :return: An OriginContainer which has True values for all origins that are True in at least one of the input
    OriginContainers
    """
    cl: ConfigLoader = ConfigLoader.instance
    result_list: List[str] = []
    for o in cl.active_origins:
        if origin_container_1.origin_is_true(origin=o) or origin_container_2.origin_is_true(origin=o):
            result_list.append(o)
    return OriginContainer(tuple(result_list))


def validate_combined_origin_name(combined_origin: str) -> None:
    """
    Raises an Exception if the input does not refer to a combined origin
    :param combined_origin: Name to check
    :return: None
    """
    cl: ConfigLoader = ConfigLoader.instance
    assert cl.database_generation_in_progress or \
           combined_origin in cl.currently_possible_combined_origins, f'Invalid name: {combined_origin}'


@functools.lru_cache()
def parse_origin_tuple(origin_tuple: Tuple[str, ...]) -> OriginContainer:
    """
    Parses and checks the tags tuple
    :param origin_tuple: Tuple that contains origins to consider
    :return: OriginContainer corresponding to the specified origin tuple
    """
    assert isinstance(origin_tuple, tuple), 'Parameter \'list_of_origins_as_tuple\' must be a tuple'
    for o in origin_tuple:
        assert type(o) == str, 'Parameter \'origins\' must be a list of strings'

    ttc: List[str] = [t for t in origin_tuple]
    result: List[str] = []
    cl: ConfigLoader = ConfigLoader.instance

    for o in cl.possible_origins:
        if o in ttc:
            result.append(o)
            ttc.remove(o)

    if not cl.database_generation_in_progress and len(ttc) != 0:
        from Project.AutoSimilarityCacheConfiguration.ConfigurationMethods import get_possible_origins
        error_message = 'Only '
        for o in get_possible_origins():
            error_message += f'\'{o}\', '
        raise Exception(error_message[:-2] + ' are allowed. Are the right combined origins set in the config file?')

    at_least_one_true: bool = False
    for o in cl.possible_origins:
        if o in result:
            at_least_one_true = True
            break
    assert at_least_one_true, 'At least one field must be set to True.'

    return OriginContainer(tuple(result))


def combined_origin_to_origin_tuple(combined_origin: str) -> Tuple[str, ...]:
    """
    :param combined_origin: A Combined Origin
    :return: Individual Origins of which the given combined origin consists of
    """
    assert isinstance(combined_origin, str), 'Parameter \'dataframe_name\' must be a string'
    validate_combined_origin_name(combined_origin)

    from Project.AutoSimilarityCacheConfiguration.ConfigurationMethods import get_possible_origins

    for part in combined_origin.split('&'):
        assert part in get_possible_origins(), 'Invalid name'
    return tuple(o for o in get_possible_origins() if o in combined_origin.split('&'))


def combined_origin_to_origin_container(combined_origin: str) -> OriginContainer:
    """
    :param combined_origin: A Combined Origin
    :return: OriginContainer containing individual Origins of which the given combined origin consists of
    """
    return OriginContainer(combined_origin_to_origin_tuple(combined_origin=combined_origin))


@functools.lru_cache(maxsize=len(ConfigLoader.instance.combined_origins_to_consider))
def __generate_df_name(origin_tuple: Tuple[str], is_rank: bool) -> str:
    origin_container = OriginContainer(origin_tuple)
    dict_name: str = ''
    cl: ConfigLoader = ConfigLoader.instance

    for o in cl.active_origins:
        if origin_container.origin_is_true(o):
            dict_name += o + '&'
    if is_rank:
        dict_name = 'rank_' + dict_name
    df_name: str = dict_name[:-1]

    if df_name.startswith('rank_'):
        assert df_name[5:] in cl.combined_origins_to_consider, \
            f'Generating name for dataframe that is not in the config file: {df_name}'
    else:
        assert df_name in cl.combined_origins_to_consider, \
            f'Generating name for dataframe that is not in the config file: {df_name}'

    return df_name


def generate_df_name(origin_container: OriginContainer, is_rank: bool) -> str:
    """
    :param origin_container: OriginContainer specifying which origins to consider for creating the name of the dataframe
    :param is_rank: True if name for a rank dataframe is to be returned, False if name of a similarity dataframe is to
    be returned
    :return: The name of the dataframe according to the specifications
    """
    return __generate_df_name(origin_container.to_origin_tuple(), is_rank)


@functools.lru_cache(maxsize=1)
def get_max_amount_of_tags_of_loaded_origins() -> int:
    """
    :return: The maximum number of tags an identifier has that come from one of the origins specified in the parameter
    "combined-origins-to-consider" in the config file.
    """
    cl: ConfigLoader = ConfigLoader.instance
    combined_origin_tuple: Tuple[str, ...] = cl.combined_origins_to_consider
    contained_origins: Set[str, ...] = set()
    for co in combined_origin_tuple:
        for o in combined_origin_to_origin_tuple(co):
            contained_origins.add(o)
    origin_container: OriginContainer = OriginContainer(origins=tuple(contained_origins))

    from Project.AutoSimilarityCacheConfiguration.DataAccess import DataAccess

    da: DataAccess = DataAccess.instance
    max_tags: int = 0
    for i in da.get_ids():
        cur_tags = da.get_tag_tuples_from_identifier(identifier=i, origin_container=origin_container)
        max_tags = max(max_tags, len(cur_tags))
    return max_tags


def origin_to_origin_container(origin: str) -> OriginContainer:
    """
    :param origin: An origin
    :return: OriginContainer where the specified origin is set to True and all other active origins are set to False
    """
    cl: ConfigLoader = ConfigLoader.instance
    assert origin in cl.active_origins
    result: List[str] = []
    for o in cl.active_origins:
        if origin == o:
            result.append(o)
    return OriginContainer(tuple(result))


def new_origin_added() -> None:
    """
    This method adapts files to fit with new origins.
    :return: None
    """
    cl: ConfigLoader = ConfigLoader.instance

    if os.path.exists(FILTER_CACHES_PATH):
        shutil.rmtree(FILTER_CACHES_PATH)

    if os.path.exists(ALL_VALUES_CHECKED + '.pkl'):
        with open(ALL_VALUES_CHECKED + '.pkl', 'rb') as f:
            all_values_checked: Dict[str, bool] = pickle.load(f)

        os.remove(ALL_VALUES_CHECKED + '.pkl')
        for o in cl.possible_dictionary_names:
            if o not in all_values_checked.keys():
                all_values_checked[o] = False
        print(all_values_checked)
        with open(ALL_VALUES_CHECKED + '.pkl', 'wb+') as f:
            pickle.dump(all_values_checked, f)

    if os.path.exists(TAG_VECTOR_CACHE_PATH):
        shutil.rmtree(TAG_VECTOR_CACHE_PATH)


def split_origin_container_to_origin_types(origin_container: OriginContainer) -> Dict[str, OriginContainer]:
    """
    :param origin_container: An OriginContainer
    :return: A dictionary containing the origin groups according to ConfigurationMethods.py as keys and tuples of
    OriginContainers as values
    """
    type_origins_dict = dict()
    for o in origin_container.to_origin_tuple():
        current_group = get_type_equally_groups(o)
        if current_group not in type_origins_dict:
            type_origins_dict[current_group] = []
        type_origins_dict[current_group].append(o)
    for k in type_origins_dict.keys():
        type_origins_dict[k] = OriginContainer(tuple(type_origins_dict[k]))
    return type_origins_dict
