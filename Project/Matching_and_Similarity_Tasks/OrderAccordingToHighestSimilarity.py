from typing import Tuple, Optional

import numpy as np

from Project.AutoSimilarityCache.Caching.TagVectorCache import TagVectorCache
from Project.AutoSimilarityCacheConfiguration.ConfigurationMethods import get_vector, get_similarity
from Project.AutoSimilarityCacheConfiguration.DataAccess import DataAccess
from Project.Utils.Misc.OriginContainer import OriginContainer


def order_according_to_highest_similarity(search_term: str, origin_container: OriginContainer,
                                          search_space: Optional[Tuple[str, ...]]):
    """
    :param search_term: The term according to which to order the collection
    :param origin_container: Origins to consider
    :param search_space: Identifiers to consider
    :return: Identifiers in the search space, ordered by the highest similarity one of its tags has to the search term
    """
    results = dict()
    da: DataAccess = DataAccess.instance
    tvc: TagVectorCache = TagVectorCache.instance
    search_term_vector: any = get_vector((search_term, 'Title'))

    if search_space is None:
        search_space = da.get_ids()

    for i in search_space:
        max_result = - np.inf
        for tt in da.get_tag_tuples_from_identifier(i, origin_container):
            tt_vector: any = tvc.get_vector_for_tag_tuple(tag_origin_tuple=tt)
            max_result = max(max_result, get_similarity(vector_origin_tuple_1=(search_term_vector, 'Title'),
                             vector_origin_tuple_2=(tt_vector, tt[1])))
        results[i] = max_result

    return tuple(r[0] for r in sorted(results.items(), key=lambda x: x[1], reverse=True))