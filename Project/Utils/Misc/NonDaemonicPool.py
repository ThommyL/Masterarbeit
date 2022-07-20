"""
NonDaemonicPool.py
"""

import multiprocessing.pool


class NonDaemonicPool(multiprocessing.pool.Pool):
    """
    Note: We can do this in this particular case, since we know that the parent threads will be able to terminate if
    and only if the children threads have finished. As an additional safety, an error would result in a path that is not
    the right length, which would throw an Exception
    Basically it is the Pool class from the multiprocessing library, except that the getter for the daemon property
    always returns False and the setter does nothing.
    """

    # noinspection PyPep8Naming,PyMissingOrEmptyDocstring
    @staticmethod
    def Process(self, *args, **kwargs):
        process = super(NonDaemonicPool, self).Process(*args, **kwargs)
        type(process).daemon = property(lambda: False, lambda: None)
        return process
