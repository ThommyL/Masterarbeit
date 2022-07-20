"""
Singleton.py
"""


class Singleton:
    """
    Wrapper that provides a typical Singleton pattern. Provides option to delete the instance, so it can be reset or
    removed from memory.
    """

    def __init__(self, cls: type):
        """
        :param cls: Type that should be wrapped with this class
        """
        assert isinstance(cls, type), 'cls argument is expected to be a class'
        self._instance = None
        self.cls = cls

    def __call__(self, *args, **kwargs):
        raise Exception('This class is a Singleton and is not meant to be called. Instead use the instance property '
                        'of this class. It will initialize the object once and then keep returning the same object '
                        'until the property is deleted (with "del class_name.instance)".')

    @property
    def instance(self):
        """
        :return: The instance of the class, whereas the instance is the same across multiple call
        """
        if self._instance is None:
            self._instance = self.cls()
        return self._instance

    @instance.deleter
    def instance(self):
        self._instance = None
