"""
Algorithm to fill spaces of a path where some artworks are given.
"""

from typing import List, Optional, Tuple, Dict

from Project.AutoSimilarityCache.Interface import get_similarities_for_id
from Project.AutoSimilarityCacheConfiguration.DataAccess import DataAccess
from Project.Matching_and_Similarity_Tasks import InputIdentifierWithoutTagsException, TooFewSamplesLeftException
from Project.Utils.ConfigurationUtils.ConfigLoader import ConfigLoader
from Project.Utils.FilterCache.FilterCache import FilterCache
from Project.Utils.Misc.OriginContainer import OriginContainer


def fill_spaces(
        to_fill: Tuple[Optional[str], ...], origin_container: OriginContainer,
        search_space: Tuple[str, ...] = None) -> Tuple[Tuple[Tuple[str, float], ...], ...]:
    """
    :param to_fill: Tuple containing identifiers and None entries
    :param origin_container: OriginContainer specifying which origins should be considered
    :param search_space: Tuplle containing identifiers that are allowed in the result
    :return: For each None entry in the input all the identifiers in the search space ordered by their rank in regards
    to filling that position.
    """

    da: DataAccess = DataAccess.instance

    if search_space is None:
        search_space = da.get_ids()

    search_space = set(search_space)
    for f in to_fill:
        if f in search_space:
            search_space.remove(f)
    search_space = tuple(search_space)

    if len(search_space) == 0:
        raise TooFewSamplesLeftException()

    for f in to_fill:
        if f is None:
            continue
        if len(da.get_tag_tuples_from_identifier(identifier=f, origin_container=origin_container)) == 0:
            raise InputIdentifierWithoutTagsException()

    fc: FilterCache = FilterCache.instance
    search_space = [s for s in search_space if fc.check_identifier(identifier=s, origin_container=origin_container)]

    similarities: List[Optional[Dict[str, float]]] = []
    for f in to_fill:
        if f is None:
            similarities.append(None)
        else:
            cl: ConfigLoader = ConfigLoader.instance
            weight_dict = cl.origin_type_group_weight_dict

            similarities.append(
                get_similarities_for_id(identifier=f, origin_container=origin_container, weight_dict=weight_dict))

    weights: List[Optional[List[float]]] = []

    processed_nones: List[int] = []
    while True:
        previous: List[Optional[Dict[str, float]]] = []
        previous_weights: List[float] = []
        following: List[Optional[Dict[str, float]]] = []
        following_weights: [List[float]] = []
        none_ind: Optional[int] = None
        for ind, s in enumerate(similarities):
            if s is None and ind not in processed_nones:
                none_ind = ind
                processed_nones.append(ind)
                break
        if none_ind is None:
            break
        for ind, s in enumerate(similarities):
            if s is None:
                continue
            if ind > none_ind:
                following.append(similarities[ind])
                following_weights.append((len(similarities) - (ind - none_ind)) / len(similarities))
            else:
                previous.append(similarities[ind])
                previous_weights.append((len(similarities) - (none_ind - ind)) / len(similarities))

        previous_weights = [w ** 2 for w in previous_weights]
        previous_weights = [w / sum(previous_weights) for w in previous_weights]

        following_weights = [w ** 2 for w in following_weights]
        following_weights = [w / sum(following_weights) for w in following_weights]

        current_weights: List[float] = previous_weights + following_weights
        weights.append(current_weights)

    # padding of weight lists
    temp_weights: List[Optional[List[float]]] = []
    for w in weights:
        new_list: List[Optional[float]] = []
        ind: int = -1
        for s in similarities:
            if s is None:
                new_list.append(None)
            else:
                ind += 1
                new_list.append(w[ind])
        temp_weights.append(new_list)
    weights = temp_weights

    temp_weights: List[Optional[List[float]]] = []
    ind: int = -1
    for f in to_fill:
        if f is None:
            ind += 1
            temp_weights.append(weights[ind])
        else:
            temp_weights.append(None)
    weights = temp_weights

    similarity_sums: List[Optional[float]] = []
    for s in similarities:
        if s is None:
            similarity_sums.append(None)
            continue
        cur_sum: float = 0
        for r in search_space:
            found: bool = True
            for ss in similarities:
                if ss is None:
                    continue
                if r not in ss.keys():
                    found = False
                    break
            if found:
                cur_sum += (1 - s[r])
        similarity_sums.append(cur_sum)

    result: List[Tuple[Tuple[str, float], ...]] = []
    highest_first: bool = False  # Note: The lowest number at the beginning

    for fill_ind, f in enumerate(to_fill):
        if f is not None:
            result.append(((f, 1),))
            continue
        weighted_similarities: Dict[str, Optional[float]] = dict()
        for r in search_space:
            found: bool = True
            for s in similarities:
                if s is None:
                    continue
                if r not in s.keys():
                    found = False
                    break
            if not found:
                continue
            weighted_similarities[r] = 0
            for sim_ind, s in enumerate(similarities):
                if s is None:
                    continue
                weighted_similarities[r] += (1 - s[r]) * weights[fill_ind][sim_ind] / similarity_sums[sim_ind]
        # noinspection PyTypeChecker
        result.append(tuple(sorted(tuple(weighted_similarities.items()), key=lambda x: x[1], reverse=highest_first)))

    return tuple(result)
