"""
Misc.py
"""
from typing import List

import numba
import numpy as np

from Project.Utils.Misc.OriginContainer import OriginContainer


def boolean_values_to_origin_container(
        des: bool, title: bool, exp: bool, icon: bool, not_icon: bool, obj_tags: bool) -> OriginContainer:
    """
    If origins are added, calls to this method have to be updated
    :param des: True if tags from descriptions should be True in the OriginContainer
    :param title: True if tags from titles should be True in the OriginContainer
    :param exp: True if tags from expert tags should be True in the OriginContainer
    :param icon: True if tags from iconclass tags should be True in the OriginContainer
    :param not_icon: True if tags from non-iconclass expert tags should be True in the OriginContainer
    :param obj_tags: Whether to consider tags from the origin "Obj".
    :return: OriginContainer according to the above specifications
    """
    result: List[str] = []
    if des:
        result.append('Des')
    if title:
        result.append('Title')
    if exp:
        result.append('Exp')
    if icon:
        result.append('Icon')
    if not_icon:
        result.append('NotIcon')
    if obj_tags:
        result.append('Obj')
    assert len(result) > 0, 'At least one value must be set to True'
    return OriginContainer(tuple(result))


@numba.jit(nopython=True)
def cosine_distance(arr_1: np.array, arr_2: np.array) -> float:
    """
    Note: For efficiency reasons, the inputs are not validated
    :param arr_1: First np.array for the equation
    :param arr_2: Second np.array for the equation
    :return: The cosine distance between the two np.arrays
    """
    return (arr_1 @ arr_2) / (np.linalg.norm(arr_1) * np.linalg.norm(arr_2))
