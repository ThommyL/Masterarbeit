"""
InvalidConfigurationException.py
"""


class InvalidConfigurationException(Exception):
    """
    This Exception is raised in ConfigurationMethods.py when an error is found by one of the configuration methods.
    """
    def __init__(self, message: str):
        self.__message = message

    def __str__(self):
        return self.__message
