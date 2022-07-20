"""
__init__.py of Matching
"""


class InputIdentifierWithoutTagsException(Exception):
    """
    This Exception is raised when at least one of the identifiers that were given as start or end points does not
    have any tags associated to it that correspond to the origins specified.
    """

    def __str__(self):
        return 'At least one of the identifiers that were given as start or end points does not have any tags ' \
               'associated to them that correspond to the origins specified.'


class TooFewSamplesLeftException(Exception):
    """
    This Exception is raised when too few samples are left to produce the requested result.
    """

    def __str__(self):
        return 'Too few samples are left to produce the requested result. Have you ' \
               'used the dynamic filtering with rules that are too strict?'
