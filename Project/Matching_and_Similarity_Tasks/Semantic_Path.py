"""
Semantic_Path.py

Note: There is one public method here:
generate_paths

Exceptions that might get raised:
TooFewSamplesLeftException
InputIdentifierWithoutTagsException
"""
import random
from typing import List, Tuple, Dict

from Project.AutoSimilarityCache.Interface import get_similarities_for_pair, \
    ensure_similarities_for_pair_are_generated
from Project.AutoSimilarityCacheConfiguration.DataAccess import DataAccess
from Project.Matching_and_Similarity_Tasks import TooFewSamplesLeftException, InputIdentifierWithoutTagsException
from Project.Utils.ConfigurationUtils.ConfigLoader import ConfigLoader
from Project.Utils.FilterCache.FilterCache import FilterCache
from Project.Utils.Misc.Misc import boolean_values_to_origin_container
from Project.Utils.Misc.NonDaemonicPool import NonDaemonicPool
from Project.Utils.Misc.OriginContainer import OriginContainer
from Project.Utils.Misc.ProcessAndThreadHandlers import ProcessHandler


def __weight_rank_dicts_together(
        similarities_1: Dict[str, int], similarities_2: Dict[str, int],
        weight_1: float, weight_2: float, search_space: Tuple[str, ...]) -> Tuple[Tuple[str, float], ...]:
    """
    :param similarities_1: A dictionary containing identifiers as keys and floats between 0 and 1 as values
    :param similarities_2: A dictionary containing identifiers as keys and floats between 0 and 1 as values
    :param weight_1: Value with which values of similarities_1 are weighted
    :param weight_2: Value with which values of similarities_2 are weighted
    :param search_space: Tuple of identifiers that should be considered
    :return: Dictionary containing best to worst match as keys in order. The Values indicate the quality of the match.
    The lower the values, the better the match.
    """
    sim_sum_1: float = 0
    sim_sum_2: float = 0

    for r in search_space:
        if r not in similarities_1.keys() or r not in similarities_2.keys():
            continue
        sim_sum_1 += (1 - similarities_1[r])
        sim_sum_2 += (1 - similarities_2[r])

    weighted_similarities: Dict[str, float] = dict()

    for r in search_space:
        if r not in similarities_1.keys() or r not in similarities_2.keys():
            continue
        pos_1: float = (1 - similarities_1[r]) * weight_1 / sim_sum_1
        pos_2: float = (1 - similarities_2[r]) * weight_2 / sim_sum_2
        weighted_similarities[r] = (pos_1 + pos_2)

    highest_first: bool = False  # Note: The lowest number at the beginning

    # noinspection PyTypeChecker
    return tuple(sorted(tuple(weighted_similarities.items()), key=lambda x: x[1], reverse=highest_first))


def __weight_based_matching(start_identifier: str, end_identifier: str, intermediate_steps: int,
                            origin_container: OriginContainer, search_space: Tuple[str, ...] = None,
                            partition_percent: int = 100) -> Tuple[str, ...]:
    """
    This method finds a path between two image identifiers by weighting the similarities of the images most similar to
    each of them, whereas the weights shift along the path.
    :param start_identifier: The string that identifies the sample in the dataframe which is the leftmost image in the
    series
    :param end_identifier: The string that identifies the sample in the dataframe which is the rightmost image in the
    series
    :param intermediate_steps: The amount of images that should be found between the start and end
    100 that specifies how many percent of the top ranking sample should be considered for the next recursive call
    :param origin_container: Which origins from tags should be considered for the search ('Des', 'Title', 'Exp', ...)
    :param search_space: Tuple of identifiers that should be considered
    :return: A tuple of identifiers containing a smooth path between start_identifier and end_identifier
    """
    cl: ConfigLoader = ConfigLoader.instance
    weight_dict = cl.origin_type_group_weight_dict

    ranks_1: Dict[str, int]
    ranks_2: Dict[str, int]
    ranks_1, ranks_2 = get_similarities_for_pair(
        id_1=start_identifier, id_2=end_identifier, origin_container=origin_container, weight_dict=weight_dict)

    weights = [(intermediate_steps - (i - 1), i) for i in range(1, intermediate_steps + 1)]
    result: List[str, ...] = [start_identifier]

    if partition_percent != 100:
        search_space = (ranked := tuple(e[0] for e in __weight_rank_dicts_together(
            similarities_1=ranks_1, similarities_2=ranks_2, weight_1=1, weight_2=1, search_space=search_space)
                                        ))[0:int(len(ranked) * partition_percent / 100)]

    weighted_results: List[List[Tuple[str, float], ...], ...] = []

    for w in weights:
        weighted_results.append(
            list(__weight_rank_dicts_together(
                similarities_1=ranks_1, similarities_2=ranks_2, weight_1=w[0], weight_2=w[1],
                search_space=search_space)))

    # Resolving ties
    updated: bool = True
    while updated:
        updated = False
        for i in range(len(weights)):
            current_i = weighted_results[i][0]
            for j in range(len(weights)):
                if i == j:
                    continue
                current_j = weighted_results[j][0]
                if current_i[0] == current_j[0] and current_i[1] >= current_j[1]:
                    del weighted_results[i][0]
                    updated = True
                    break
            if updated:
                break

    for w in weighted_results:
        current_result = [r[0] for r in w]
        for r in current_result:
            if r not in result and r != end_identifier:
                result.append(r)
                break
    result.append(end_identifier)
    return tuple(result)


def __matching(
        start_identifier: str, end_identifier: str, intermediate_steps: int, origin_container: OriginContainer,
        partition_percent: int = 100) -> Tuple[str, ...]:
    """
    This method takes two image identifiers as input and finds a path through other images of the dataset according to
    parameterization between them.
    :param start_identifier: The string that identifies the sample in the dataframe which is the leftmost image in the
    series
    :param end_identifier: The string that identifies the sample in the dataframe which is the rightmost image in the
    series
    :param intermediate_steps: The amount of images that should be found between the start and end
    :param origin_container: Which origins from tags should be considered for the search ('Des', 'Title', 'Exp', ...)
    :param partition_percent: Int between int(intermediate_steps / len(cd.get_ids())) ** power * 100 + 1 and
    100 that specifies how many percent of the top ranking sample should be considered for the next recursive call
    :return: A tuple of identifiers containing a smooth path between start_identifier and end_identifier
    """

    def matching_inner(inner_start_identifier: str, inner_end_identifier: str, inner_intermediate_steps: int,
                       inner_result: Dict[int, str], inner_search_space: Tuple[str, ...],
                       index_offset: int = 0) -> Dict[int, str]:
        """
        :param inner_start_identifier: The string that identifies the sample in the dataframe which is the leftmost
        image in the current part of the series
        :param inner_end_identifier: The string that identifies the sample in the dataframe which is the rightmost
        image in the current part of the series
        :param inner_intermediate_steps: The amount of images that should be found between the inner_start_identifier
        and inner_end_identifier
        :param inner_result: Already found identifiers
        :param inner_search_space: The identifiers that should be checked
        :param index_offset: An offset to calculate the position in the result dictionary at which to write results
        :return: The updated result dictionary
        """

        def update_result_with_rank_dict(cur: Tuple[str, ...], res: Dict[int, str], pos: int) -> Dict[int, str]:
            """
            :param cur: Currently calculated results with which the result dictionary should be updated with
            :param res: The current result
            :param pos: The position on the path for which to write the result for
            :return: The updated result dictionary
            """
            assert pos not in res.keys(), 'Overwriting entry in result.'
            config_loader: ConfigLoader = ConfigLoader.instance
            id_set: Tuple[str, ...] = tuple(e[1] for e in res.items())
            updated = False
            for r in cur:
                if r not in id_set:
                    if config_loader.matching_debug_output_enabled:
                        print(f'Matching Algorithm: Found Identifier {r} with recursive matching.')
                    res[pos] = r
                    updated = True
                    break
            if not updated:
                raise TooFewSamplesLeftException()
            return res

        def update_result_with_weight_based_matching(res, start: str, end: str, intermediate: int,
                                                     search: Tuple[str, ...]) -> Dict[int, str]:
            """
            :param res: The current result dictionary
            :param start: The string that identifies the sample in the dataframe which is the leftmost image in the
            current part of the series
            :param end: The string that identifies the sample in the dataframe which is the rightmost image in the
            current part of the series
            :param intermediate: The amount of images that should be found between the inner_start_identifier and
            inner_end_identifier
            :param search: The identifiers that should be checked
            :return: The updated result dictionary
            """
            smooth_result: Tuple[str, ...] = __weight_based_matching(
                start_identifier=start, end_identifier=end, intermediate_steps=intermediate,
                origin_container=origin_container,
                search_space=tuple(se for se in search if se not in [i[1] for i in res.items()]),
                partition_percent=partition_percent)[1:-1]
            for i in range(2, len(smooth_result) + 2):
                assert index_offset + i not in res.keys(), 'Overwriting entry in result.'
                res[index_offset + i] = smooth_result[i - 2]
            return res

        cl: ConfigLoader = ConfigLoader.instance

        if not cl.matching_experimental_mode:
            return update_result_with_weight_based_matching(
                res=inner_result, start=inner_start_identifier, end=inner_end_identifier,
                intermediate=inner_intermediate_steps, search=tuple(search_space))

        middle: float = (2 + inner_intermediate_steps) / 2
        left_middle: int = int(middle) + 1
        right_middle: int = int(middle) + 1
        current_results: List[str, ...] = []
        if inner_intermediate_steps % 2 == 0:
            left_middle = right_middle - 1
            greater_weight = right_middle / middle
            to_skip = tuple(i[1] for i in inner_result.items())

            weight_dict = cl.origin_type_group_weight_dict

            ranks_1: Dict[str, int]
            ranks_2: Dict[str, int]
            ranks_1, ranks_2 = get_similarities_for_pair(id_1=inner_start_identifier, id_2=inner_end_identifier,
                                                         origin_container=origin_container, weight_dict=weight_dict)
            for s in to_skip:
                if s in ranks_1:
                    del ranks_1[s]
                if s in ranks_2:
                    del ranks_2[s]

            left_results: Tuple[Tuple[str, float], ...] = __weight_rank_dicts_together(
                similarities_1=ranks_1, similarities_2=ranks_2, weight_1=greater_weight, weight_2=1,
                search_space=inner_search_space)

            right_results: Tuple[Tuple[str, float], ...] = __weight_rank_dicts_together(
                similarities_1=ranks_1, similarities_2=ranks_2, weight_1=1, weight_2=greater_weight,
                search_space=inner_search_space)

            # Resolving ties
            if left_results[0][0] == right_results[0][0]:
                if left_results[0][1] >= right_results[0][1]:
                    right_results = right_results[1:]
                else:
                    left_results = left_results[1:]

            inner_result: Dict[int, str] = update_result_with_rank_dict(
                cur=tuple(r[0] for r in left_results), res=inner_result, pos=left_middle + index_offset)

            inner_result: Dict[int, str] = update_result_with_rank_dict(
                cur=tuple(r[0] for r in right_results), res=inner_result, pos=right_middle + index_offset)

            to_skip: Tuple[str, ...] = tuple(i[1] for i in inner_result.items())

            for s in to_skip:
                if s in ranks_1:
                    del ranks_1[s]
                if s in ranks_2:
                    del ranks_2[s]

            if partition_percent != 100:
                current_results: Tuple[str, ...] = tuple(r[0] for r in __weight_rank_dicts_together(
                    similarities_1=ranks_1, similarities_2=ranks_2, weight_1=1, weight_2=1,
                    search_space=tuple(search_space)))
        else:
            middle: int = int(middle) + 1

            weight_dict = cl.origin_type_group_weight_dict

            ranks_1: Dict[str, float]
            ranks_2: Dict[str, float]
            ranks_1, ranks_2 = get_similarities_for_pair(
                id_1=inner_start_identifier, id_2=inner_end_identifier, origin_container=origin_container,
                weight_dict=weight_dict)

            for s in inner_result.items():
                if s[1] in ranks_1.keys():
                    del ranks_1[s[1]]
                if s[1] in ranks_2.keys():
                    del ranks_2[s[1]]
            current_results: Tuple[str, ...] = tuple(r[0] for r in __weight_rank_dicts_together(
                similarities_1=ranks_1, similarities_2=ranks_2, weight_1=1, weight_2=1,
                search_space=tuple(search_space)))

            inner_result: Dict[int, str] = update_result_with_rank_dict(cur=current_results, res=inner_result,
                                                                        pos=int(middle + index_offset))
        if inner_intermediate_steps <= 2:
            return inner_result
        else:
            new_intermediate_steps: int = int(inner_intermediate_steps / 2)
            right_index_offset: int = 0
            if inner_intermediate_steps % 2 == 0:
                new_intermediate_steps -= 1
                right_index_offset = 1
            if partition_percent != 100:
                inner_search_space: Tuple[str, ...] = \
                    tuple(r for r in current_results[0:int(len(current_results) / (100 / partition_percent))])
            left_first = random.randint(0, 1) == 0

            def recursive_call_left():
                return matching_inner(
                    inner_start_identifier=inner_start_identifier, inner_end_identifier=inner_result[left_middle],
                    inner_intermediate_steps=new_intermediate_steps, inner_result=inner_result,
                    inner_search_space=inner_search_space, index_offset=index_offset)

            def recursive_call_right():
                return matching_inner(
                    inner_start_identifier=inner_result[right_middle], inner_end_identifier=inner_end_identifier,
                    inner_intermediate_steps=new_intermediate_steps,
                    inner_result=inner_result, inner_search_space=inner_search_space,
                    index_offset=int(index_offset + middle - 1 + right_index_offset))
            if left_first:
                inner_result: Dict[int, str] = recursive_call_left()
                inner_result: Dict[int, str] = recursive_call_right()
            else:
                inner_result: Dict[int, str] = recursive_call_right()
                inner_result: Dict[int, str] = recursive_call_left()
        return inner_result

    fc: FilterCache = FilterCache.instance
    search_space: List[str, ...] = list(fc.get_filtered_identifiers(origin_container=origin_container))

    result: Dict[int, str] = dict()
    result[1] = start_identifier
    result[intermediate_steps + 2] = end_identifier

    # Note: Could have already been filtered out by dynamic filtering:
    if start_identifier in search_space:
        search_space.remove(start_identifier)
    if end_identifier in search_space:
        search_space.remove(end_identifier)

    result: Dict[int, str] = matching_inner(
        inner_start_identifier=start_identifier, inner_end_identifier=end_identifier,
        inner_intermediate_steps=intermediate_steps, inner_result=result, inner_search_space=tuple(search_space))

    if len(result) != intermediate_steps + 2:
        raise TooFewSamplesLeftException(
            f'Length of results is {len(result)}, but was expected to be {intermediate_steps + 2}. '
            f'For this calculation you have set the partition_percent parameter to {partition_percent} and dynamic is '
            f'{"in" if fc.is_active() else ""}active. You may raise the partition_percent parameter to 100 or change '
            f'the dynamic filtering.'
        )

    for ind in range(1, intermediate_steps + 3):
        assert ind in result.keys(), f'Could not find entry for position {ind}'

    path: Tuple[str, ...] = tuple(result[ind] for ind in range(1, len(result) + 1))
    return path


def __thread_handled_matching_along_path(
        key_point_identifiers: Tuple[str, ...], distance_between_key_points: int, origin_container: OriginContainer,
        partition_percent: int = 100, thread_handler_automatic_parameter=None) -> Tuple[str, ...]:
    """
    A callback method that is meant to be called by the ThreadHandler class only. This method exists so that a wrapping
    method from the ThreadHandler class can assign threads to this method.
    :param key_point_identifiers: Identifiers in this tuple are treated as evenly spaced images through which the
    generated path will lead
    :param distance_between_key_points: Space between individual key points
    :param origin_container: Which origins from tags should be considered for the search ('Des', 'Title', 'Exp', ...)
    :param partition_percent: Int between int(intermediate_steps / len(cd.get_ids())) ** power * 100 + 1 and
    100 that specifies how many percent of the top ranking sample should be considered for the next recursive call
    :param thread_handler_automatic_parameter: This parameter will be set automatically by the ThreadHandler class
    :return: A tuple of identifiers containing a smooth path between start_identifier and end_identifier
    """
    path: List[str, ...] = [key_point_identifiers[0]]
    params: Tuple[Tuple[any, any, int, OriginContainer, int], ...] = tuple((
                                                                               key_point_identifiers[i],
                                                                               key_point_identifiers[i + 1],
                                                                               distance_between_key_points,
                                                                               origin_container, partition_percent) for
                                                                           i in range(len(key_point_identifiers) - 1))
    result: Tuple[Tuple[str, ...], ...]

    cl: ConfigLoader = ConfigLoader.instance
    if cl.data_generation_mode_is_test and thread_handler_automatic_parameter > 0:
        with NonDaemonicPool(thread_handler_automatic_parameter) as p:
            result = tuple(p.starmap(__matching, params))
    else:
        result = tuple(__matching(*p) for p in params)
    for r in result:
        current_path = r[1:-1]
        for c in current_path:
            path.append(c)

    path.append(key_point_identifiers[len(key_point_identifiers) - 1])
    return tuple(path)


def __recursive_matching_along_path(
        key_point_identifiers: Tuple[str, ...], distance_between_key_points: int, origin_container: OriginContainer,
        partition_percent: int = 100) -> Tuple[str, ...]:
    """
    This method summarizes the parameters into args and kwargs and instructs the ThreadHandler class to carry out the
    __thread_handled_matching_along_path method with this parameterization.
    :param key_point_identifiers: Identifiers in this tuple are treated as evenly spaced images through which the
    generated path will lead
    :param distance_between_key_points: Space between individual key points
    :param origin_container: Which origins from tags should be considered for the search ('Des', 'Title', 'Exp', ...)
    :param partition_percent: Int between int(intermediate_steps / len(cd.get_ids())) ** power * 100 + 1 and
    100 that specifies how many percent of the top ranking sample should be considered for the next recursive call
    :return: A tuple of identifiers containing a smooth path between start_identifier and end_identifier
    """
    assert len(set(key_point_identifiers)) == len(key_point_identifiers), 'key_points are not unique.'
    args: Tuple[Tuple[str, ...], int, OriginContainer] = (key_point_identifiers, distance_between_key_points,
                                                          origin_container)
    kwargs: Dict[str, any] = {'partition_percent': partition_percent}
    ph: ProcessHandler = ProcessHandler.instance
    result = ph.exec_in_pool(function=__thread_handled_matching_along_path,
                             maximum=len(key_point_identifiers) - 1, args=args, kwargs=kwargs)
    ph.release_cores()
    return result


def __thread_handled_recursive_matching_along_path(
        for_input: Tuple[Tuple[str, ...], ...], distance_between_key_points: int, origin_container: OriginContainer,
        partition_percent: int = 100, thread_handler_automatic_parameter=None) -> Tuple[Tuple[str, ...], ...]:
    """
    A callback method that is meant to be called by the ThreadHandler class only. This method exists so that a wrapping
    method from the ThreadHandler class can assign threads to this method.
    :param for_input: A tuple containing tuples that contain the start and end (and possibly intermediate) images for
    which the path should be generated
    :param distance_between_key_points: Space between individual key points
    :param origin_container: Which origins from tags should be considered for the search ('Des', 'Title', 'Exp', ...)
    :param partition_percent: Int between int(intermediate_steps / len(cd.get_ids())) ** power * 100 + 1 and
    100 that specifies how many percent of the top ranking sample should be considered for the next recursive call
    :param thread_handler_automatic_parameter: This parameter will be set automatically by the ThreadHandler class
    :return: A tuple of identifiers containing a smooth path between start_identifier and end_identifier
    """
    params: Tuple[Tuple[Tuple[str, ...], int, OriginContainer, int], ...] = tuple(
        (cur, distance_between_key_points, origin_container, partition_percent) for cur in
        for_input)
    cl: ConfigLoader = ConfigLoader.instance
    if cl.data_generation_mode_is_test and thread_handler_automatic_parameter > 0:
        with NonDaemonicPool(thread_handler_automatic_parameter) as p:
            return tuple(p.starmap(__recursive_matching_along_path, params))
    else:
        return tuple(__recursive_matching_along_path(*p) for p in params)


def __generate_paths_unrestricted(
        for_input: Tuple[Tuple[str, ...], ...], distance_between_key_points: int, origin_container: OriginContainer,
        partition_percent=100) -> Tuple[Tuple[str, ...], ...]:
    """
    A method that provides an interface to all path generating tasks. In contrast to the method 'generate_paths' it
    is very open and leaves room for experimentation, but also for mistakes. Therefore it is not meant to be used in
    later stages of the project.
    :param for_input: A tuple containing tuples that contain the start and end (and possibly intermediate) images for
    which the path should be generated
    :param distance_between_key_points: Space between individual key points
    :param origin_container: Which origins from tags should be considered for the search ('Des', 'Title', 'Exp', ...)
    :param partition_percent: Parameter that regulates how many samples are discarded. How this mechanism works and
    the effect a given value of this parameter has strongly depends on whether the search is done recursively or
    linearly.
    :return: A tuple of identifiers containing a smooth path between start_identifier and end_identifier.
    """
    args: Tuple[Tuple[Tuple[str, ...], ...], int, origin_container] = (for_input, distance_between_key_points,
                                                                       origin_container)
    kwargs: Dict[str, any] = {
        'partition_percent': partition_percent}
    ph: ProcessHandler = ProcessHandler.instance
    result = ph.exec_in_pool(
        function=__thread_handled_recursive_matching_along_path, maximum=len(for_input), args=args, kwargs=kwargs)
    ph.release_cores()
    return result


def generate_path(start: str, end: str, intermediate_steps: int, title_tags: bool = True, exp_tags: bool = True,
                  icon_tags=False, not_icon_tags=False, obj_tags=False) -> Tuple[str, ...]:
    """
    The one public function that provides an interface for all path-generating tasks.
    :param start: The identifier of the artwork from which to start the path
    :param end: The identifier of the artwork at which to end the path
    :raises InputIdentifierWithoutTagsException if start or end has no tags associated with the specified origins.
    :raises TooFewSamplesLeftException if there are not enough identifiers left in the search space to somplete the
    matching. (E.g. if to many of them are filtered with the dynamic filtering)
    :param intermediate_steps: The amount of images that should be found between each start and end identifier tuple
    :param title_tags: Whether to consider tags from the origin "Title".
    :param exp_tags: Whether to consider tags from the origin "Exp".
    :param icon_tags: Whether to consider tags from the origin "Icon".
    This is meant to be used only when generating paths for the iconclass measure.
    :param not_icon_tags: Whether to consider tags from the origin "NotIcon".
    This is meant to be used only when generating paths for the iconclass measure.
    :param obj_tags: Whether to consider tags from the origin "Obj".
    :return: A tuple of tuple containing the paths corresponding to the identifier pairs of the parameter 'for_input'
    """
    da: DataAccess = DataAccess.instance
    assert isinstance(intermediate_steps, int), 'Parameter \'intermediate_steps\' must be an int'
    assert da.is_valid_identifier(start), f'Could not find identifier {start} in dataset.'
    assert da.is_valid_identifier(end), f'Could not find identifier {end} in dataset.'
    assert intermediate_steps > 0, 'Parameter \'intermediate_steps\' must be at least 1.'
    assert intermediate_steps <= 100, 'Parameter \'intermediate_steps\' must be at most 100.'
    assert title_tags in [True, False], 'Parameter \'title_tags\' must be boolean.'
    assert exp_tags in [True, False], 'Parameter \'exp_tags\' must be boolean.'
    assert not_icon_tags in [True, False], 'Parameter \'not_icon_tags\' must be boolean.'
    assert obj_tags in [True, False], 'Parameter \'obj_tags\' must be boolean.'
    assert title_tags or exp_tags or icon_tags or not_icon_tags or obj_tags, \
        'At least one of parameters \'title_tags\', \'exp_tags\', \'iconclass_tags\' or \'not_icon_tags\' ' \
        'or \'obj_tags\' must be set to True'
    origin_container = boolean_values_to_origin_container(des=False, title=title_tags, exp=exp_tags,
                                                          icon=icon_tags, not_icon=not_icon_tags, obj_tags=obj_tags)
    cl: ConfigLoader = ConfigLoader.instance

    for i in (start, end):
        if len(da.get_tag_tuples_from_identifier(i, cl.get_origin_container_with_active_origins())) == 0:
            raise InputIdentifierWithoutTagsException(f'Identifier has no tags of any of the specified origins: {i}.')

    fc: FilterCache = FilterCache.instance

    if fc.nr_of_identifiers_left(origin_container=origin_container) <= intermediate_steps + 2:
        raise TooFewSamplesLeftException()

    if cl.data_generation_mode_is_test:  # In order to use threads optimally, generate start and end points beforehand
        ensure_similarities_for_pair_are_generated(id_1=start, id_2=end, origin_container=origin_container)

    result = __generate_paths_unrestricted(for_input=((start, end),), distance_between_key_points=intermediate_steps,
                                           origin_container=origin_container)[0]

    return result
