"""
Nlp.py
"""

import spacy

from Project.Utils.Misc.Singleton import Singleton


@Singleton
class NLP:
    """
    Singleton pattern that loads the spacy Language model 'de_core_news_lg'
    """

    def __init__(self):
        # Note that spacy is not loaded in the GPU version, as it would require a significant communication overhead.
        # Also, the used model is optimized for the use on CPU.
        self.__nlp = spacy.load('de_core_news_lg')

    @property
    def nlp(self) -> spacy.language:
        """
        :return: Spacy Language model 'de_core_news_lg'
        """
        return self.__nlp
