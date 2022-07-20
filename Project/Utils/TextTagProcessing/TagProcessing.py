"""
TagProcessing.py
"""

from typing import List, Tuple, Set

from Project.AutoSimilarityCacheConfiguration.DataAccess import DataAccess


def __check_string_arrays(test_for: Tuple[str, ...], search_in: Tuple[str, ...]) -> bool:
    """
    :param test_for: The string array for which to look for
    :param search_in: The string array in which to search
    :return: True if search in starts with test_for
    """
    if len(search_in) < len(test_for):
        return False
    for t, s in zip(test_for, search_in):
        if t != s:
            return False
    return True


def filter_doubles(identifier: str, tag_tuples: Tuple[Tuple[any, str]]) -> Tuple[Tuple[any, str]]:
    """
    Some tags appear twice, since e.g. when the same tag is generated from different origins. In this case only one
    instance is kept.
    Some tags are contained in each other. This method therefore also filters out the tags that are (on a per-word
    basis) fully contained in other tags. Example: Of the three tags "Filtering is great", "Filtering is", "Filter"
    only "Filtering is" would be filtered out.
    :param identifier
    :param tag_tuples: The tuples that should be filtered.
    :return: The filtered tuples
    """
    assert isinstance(tag_tuples, tuple), 'Parameter \'tag_tuples\' must be a tuple'
    if len(tag_tuples) == 0:
        return tuple()
    # Note: only one element is checked for efficiency reasons since this function is called many times
    assert isinstance(tag_tuples[0], tuple), 'List elements must be Tuples'
    assert isinstance(tag_tuples[0][0], str), 'Tuple elements must be str'
    assert isinstance(tag_tuples[0][1], str), 'Tuple elements must be str'

    to_delete_double: Set[str] = set()
    to_delete_beginning: Set[str] = set()
    da: DataAccess = DataAccess.instance

    for i in range(len(tag_tuples)):
        cur_tag_tuple: Tuple[str, str] = tag_tuples[i]
        cur_tag: str = cur_tag_tuple[0].lower()
        ct_tuple: Tuple[str, ...] = tuple(cur_tag.lower().split(' '))

        for j in range(len(tag_tuples)):
            if i == j:
                continue
            compare_to_tag_tuple: Tuple[str, str] = tag_tuples[j]
            compare_to_tag: str = compare_to_tag_tuple[0].lower()
            ctt_tuple: Tuple[str, ...] = tuple(compare_to_tag.lower().split(' '))

            # Filter a tag if it appears twice, no matter the origin
            if ctt_tuple == ct_tuple:
                if da.get_weight_for_identifier_tag_tuple(
                        identifier=identifier, tag_origin_tuple=compare_to_tag_tuple) > \
                        da.get_weight_for_identifier_tag_tuple(identifier=identifier, tag_origin_tuple=cur_tag_tuple):
                    to_delete_double.add(' '.join(ct_tuple))
                else:
                    to_delete_double.add(' '.join(ctt_tuple))
                continue

            # Filter e.g. "Tag" if "Tags are great" is present
            for k in range(min(len(ctt_tuple), len(ct_tuple))):
                if ctt_tuple[k] == ct_tuple[k]:
                    continue
                elif ctt_tuple[k].startswith(ct_tuple[k]):
                    to_delete_beginning.add(' '.join(ct_tuple))
                    break
                else:
                    break
    # For efficiency reasons a set is used here, after all the code will be executed many thousands of times
    encountered: Set[str] = set()
    cleaned: List[Tuple[any, str]] = []
    for entry in [t for t in tag_tuples]:
        if entry[0].lower() in to_delete_double and entry[0].lower() in encountered:
            continue
        else:
            cleaned.append(entry)
            encountered.add(entry[0].lower())
    tag_tuples = tuple(cleaned)

    # Add the to_delete_beginning to the set, so that we need to delete only twice instead of 3 times
    to_delete_overlap: Set[str] = to_delete_beginning
    for i in range(len(tag_tuples)):
        cur_tag_tuple: Tuple[str, str] = tag_tuples[i]
        cur_tag: str = cur_tag_tuple[0].lower()
        ct_tuple: Tuple[str, ...] = tuple(cur_tag.lower().split(' '))

        for j in range(len(tag_tuples)):
            if i == j:
                continue
            compare_to_tag_tuple = tag_tuples[j]
            compare_to_tag = compare_to_tag_tuple[0].lower()
            ctt_tuple = tuple(compare_to_tag.lower().split(' '))
            # Filter e.g. "I am a tag" if "I am a tag too" is present
            for k in range(len(ctt_tuple)):
                if __check_string_arrays(ct_tuple, ctt_tuple[k:]):
                    to_delete_overlap.add(' '.join(ct_tuple))

    return tuple(t for t in tag_tuples if t[0].lower() not in to_delete_overlap)
