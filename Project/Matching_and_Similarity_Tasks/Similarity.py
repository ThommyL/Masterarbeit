"""
Similarity.py
"""

from typing import Tuple, Dict

from Project.AutoSimilarityCache.Interface import get_similarities_for_id, \
    filter_and_optimize_data_tuples
from Project.AutoSimilarityCacheConfiguration.DataAccess import DataAccess
from Project.Utils.ConfigurationUtils.ConfigLoader import ConfigLoader
from Project.Utils.FilterCache.FilterCache import FilterCache
from Project.Utils.Misc.OriginContainer import OriginContainer


def get_most_similar_artworks_and_their_tags(identifier: str, origin_container: OriginContainer, length: int
                                             ) -> Tuple[Tuple[Tuple[any, str], ...],
                                                        Dict[str, Tuple[Tuple[any, str], ...]]]:
    """
    :param identifier: Identifier of the artwork for which to find the most similar artworks
    :param origin_container: Origins from which tags should be considered
    :param length: The amount of results that should be returned
    :return: Tuple with first element being the tags of the given identifier that were considered in the matching
    process (see method "filter_and_optimize_tag_tuples"), and the second element being a Tuple containing the
    best matches with their tags that were considered for the matching.
    """
    da: DataAccess = DataAccess.instance
    da.is_valid_identifier(identifier=identifier)
    assert isinstance(length, int), 'Parameter \'length\' must be an int'
    assert 0 < length <= 100, 'Parameter \'length\' must be between 1 and 100'

    fc: FilterCache = FilterCache.instance
    cl: ConfigLoader = ConfigLoader.instance

    weight_dict = cl.origin_type_group_weight_dict

    similarities = get_similarities_for_id(
        identifier=identifier, origin_container=origin_container, weight_dict=weight_dict)
    similarities = [r[0] for r in sorted(similarities.items(), key=lambda x: x[1])]

    results: Dict[str, Tuple[Tuple[any, str], ...]] = dict()

    while len(similarities) > 0 and len(results.keys()) < length:
        current = similarities.pop()
        if fc.check_identifier(identifier=current, origin_container=origin_container):
            results[current] = filter_and_optimize_data_tuples(identifier=current, origin_container=origin_container)
    return filter_and_optimize_data_tuples(identifier=identifier, origin_container=origin_container), results
