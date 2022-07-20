"""
OriginContainer.py
"""

from typing import Tuple


class OriginContainer:
    """
    This class exists in order to make modifications in the future easier.
    Note: No checks are done in this class for efficiency reasons.
    """

    def __init__(self, origins: Tuple[str, ...]):
        """
        This class exists to make the program easier to adapt in the future.
        :param origins: Tuple containing the names of the origins.
        """
        assert isinstance(origins, tuple)
        self.__origins = origins

    def __str__(self):
        return str(self.__origins)

    def __hash__(self):
        return self.to_origin_tuple().__hash__()

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()

    @property
    def number_of_true_origins(self) -> int:
        """
        :return: The number of origins in this origin container that are set to True
        """
        return len(self.__origins)

    def origin_is_true(self, origin: str):
        """
        :param origin: Origin for which to check
        :return: True if origin is True, False otherwise
        """
        return origin in self.__origins

    def to_origin_tuple(self) -> Tuple[str, ...]:
        """
        :return: An origin tuple corresponding to this OriginContainer
        """
        return tuple(self.__origins)
