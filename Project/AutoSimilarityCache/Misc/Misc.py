"""
Misc.py
"""

import functools
from typing import Tuple, List

from tqdm import tqdm

from Project.AutoSimilarityCache.Caching import IDENTIFIER_COLUMN_NAME
from Project.AutoSimilarityCacheConfiguration.ConfigurationMethods import filter_in_advance, \
    check_additional_constraints
from Project.AutoSimilarityCacheConfiguration.DataAccess import DataAccess
from Project.Utils.Misc.OriginContainer import OriginContainer


@functools.lru_cache()
def filter_and_optimize_data_tuples(identifier: str, origin_container: OriginContainer) -> Tuple[Tuple[any, str], ...]:
    """
    :param identifier: Identifier of which to return the filtered tags of.
    :param origin_container: OriginContainer specifying which tags should be considered
    :return: Filtered input
    """
    return filter_in_advance(identifier=identifier, origin_container=origin_container)


def split_list_into_equal_parts(
        to_split: Tuple[any, ...], nr_of_parts: int) -> Tuple[Tuple[Tuple[any, str]]]:
    """
    :param to_split: The tuple that should be split
    :param nr_of_parts: The number of parts the tuple should be split into
    :return: Tuple split into nr_of_parts roughly equally long parts
    """
    assert isinstance(to_split, tuple), 'Parameter \'to_split\' must be a tuple'
    assert isinstance(nr_of_parts, int), 'Parameter \'nr_of_parts\' must be an int'
    chunk_size: int = int(len(to_split) / nr_of_parts) + 1
    list_of_tags: List[Tuple[Tuple[any, str]]] = []
    one_dim_list: List[Tuple[any, str]] = []
    last: int = 0
    for ind, elem in enumerate(tqdm(to_split, desc='Splitting up input')):
        one_dim_list.append(elem)
        if chunk_size == (ind + 1) - last:
            last = ind + 1
            list_of_tags.append(tuple(one_dim_list))
            one_dim_list = []
    if len(one_dim_list) > 0:
        list_of_tags.append(tuple(one_dim_list))
    assert len(list_of_tags) == nr_of_parts
    return tuple(list_of_tags)


def check_constraints() -> None:
    """
    This method checks restraints given by the implementation of this program or by the configuration methods.
    :return: None
    """
    da: DataAccess = DataAccess.instance
    for i in da.get_ids():
        assert i != IDENTIFIER_COLUMN_NAME, f'The name {IDENTIFIER_COLUMN_NAME} cannot be used as an identifier.'
    check_additional_constraints()
