"""
IconclassCache.py
"""

import functools
import os
import pickle
import sqlite3
from sqlite3 import Connection, Cursor
from typing import List, Tuple, Set, Optional

import iconclass
import pandas as pd
from tqdm import tqdm

from Project.Utils.IconclassCache import ICONCLASS_CATEGORY_TO_NAME_PKL, \
    ICONCLASS_CACHE_GENERATION_WAS_SUCCESSFUL_PKL, ICONCLASS_NUMBER_OF_TRIES_PKL
from Project.Utils.Misc.ProcessAndThreadHandlers import ProcessHandler
from Project.Utils.Misc.Singleton import Singleton


class NumberOfTriesExceededException(Exception):
    """
    This Exception should be raised if the initialization failed for at least 3 times.
    """

    def __str__(self):
        return 'Initialization does not work anymore. This process is dependent on external resources.' \
               ' It may be the case that these are either not available anymore or that they have ' \
               'changed. This Exception was raised in order to prevent further attempts. If you ' \
               f'want to try again anyway you need to delete {ICONCLASS_NUMBER_OF_TRIES_PKL}.pkl'


class IconclassCacheGenerationFailedException(Exception):
    """
    This Exception should be raised for the first two times an error occurs during initialization. After that the
    NumberOfTriesExceededException should be raised.
    """

    def __str__(self):
        return 'An error occoured. If the IconclassCache was not initialized before, please check your internet ' \
               'connection, since this class requires external resources and try again.'


@Singleton
class IconclassCache:
    """
    This class provides a way to compare the results of the path finding algorithm with the iconclass system.
    :raises IconclassCacheGenerationFailedException: This class loads external resources. For the first two times an
    error occurs during the loading or processing of the loaded data, this Exception is raised.
    :raises NumberOfTriesExceedsException: This class loads external resources. If an error occurs during the loading
    or processing of the loaded data for the third time, this exception is raised.
    """

    def __init__(self):
        if os.path.exists(ICONCLASS_CATEGORY_TO_NAME_PKL + '.pkl'):
            self.__df = pd.read_pickle(ICONCLASS_CATEGORY_TO_NAME_PKL + '.pkl')
        else:
            with open(ICONCLASS_CACHE_GENERATION_WAS_SUCCESSFUL_PKL + '.pkl', 'wb+') as f:
                pickle.dump(False, f)
            ph: ProcessHandler = ProcessHandler.instance
            while True:
                try:
                    if not ph.acquire_data_generation_lock():
                        if os.path.exists(ICONCLASS_CATEGORY_TO_NAME_PKL + '.pkl'):
                            # if this was generated in the meantime
                            self.__df: pd.DataFrame = pd.read_pickle(ICONCLASS_CATEGORY_TO_NAME_PKL + '.pkl')
                            break
                        if self.__get_number_of_tries__() >= 3:
                            raise NumberOfTriesExceededException()
                        self.__increment_number_of_tries__()

                        # There is no explicit way to make the library fetch the database
                        _ = iconclass.fetch_from_db('')

                        db: Connection = sqlite3.connect(os.environ.get("ICONCLASS_DB_LOCATION", "iconclass.sqlite"))
                        cursor: Cursor = db.cursor()
                        sql: str = 'SELECT texts.ref FROM texts'
                        cursor.execute(sql)
                        text_refs: Set[str] = set(r[0] for r in cursor.fetchall() if r[0] != '')

                        result_set: Set[Tuple[str, str]] = set()

                        for elem in tqdm(text_refs, desc='Loading iconclass categories and texts'):
                            cursor: Cursor = db.cursor()
                            sql: str = 'SELECT notations.notation, texts.text ' \
                                       'FROM texts LEFT JOIN notations ON notations.id = texts.ref ' \
                                       'WHERE texts.ref = ? and texts.type == 0 and texts.language == "de"'
                            cursor.execute(sql, (elem,))
                            cur_result: List = cursor.fetchall()

                            if len(cur_result) != 0:
                                for c in cur_result:
                                    result_set.add(c)

                        df: pd.DataFrame = pd.DataFrame(result_set, columns=['Category', 'Text'])

                        with open(ICONCLASS_CACHE_GENERATION_WAS_SUCCESSFUL_PKL + '.pkl', 'wb') as f:
                            pickle.dump(True, f)
                        df.copy().to_pickle(ICONCLASS_CATEGORY_TO_NAME_PKL + '.pkl')
                        self.__df = df

                # It is unknown in what way this might fail if the external resources change
                except Exception as e:
                    try:
                        raise e
                    finally:
                        raise IconclassCacheGenerationFailedException()
                break
            ph.release_data_generation_lock()

    @staticmethod
    @functools.lru_cache()
    def was_initiated_successfully() -> bool:
        """
        :return: True if the initialization of the cache was successful, False otherwise
        """
        with open(ICONCLASS_CACHE_GENERATION_WAS_SUCCESSFUL_PKL + '.pkl', 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def __get_number_of_tries__() -> int:
        """
        :return: The number of times the initialization was tried.
        """
        if os.path.exists(ICONCLASS_NUMBER_OF_TRIES_PKL + '.pkl'):
            with open(ICONCLASS_NUMBER_OF_TRIES_PKL + '.pkl', 'rb') as f:
                return pickle.load(f)
        else:
            return 0

    @staticmethod
    def __increment_number_of_tries__() -> None:
        """
        Increases the number of tries by one and writes the result to a file.
        :return: None
        """
        number_of_tries: int = 0
        if os.path.exists(ICONCLASS_NUMBER_OF_TRIES_PKL + '.pkl'):
            with open(ICONCLASS_NUMBER_OF_TRIES_PKL + '.pkl', 'rb') as f:
                number_of_tries = pickle.load(f)
        with open(ICONCLASS_NUMBER_OF_TRIES_PKL + '.pkl', 'wb+') as f:
            pickle.dump(number_of_tries, f)

    def text_to_category(self, text: str) -> Optional[str]:
        """
        :param text: German text that is used to label an artwork
        :return: Iconclass category that is described by the text input. None in case none was found.
        """
        row: pd.DataFrame = self.__df.loc[(self.__df['Text'] == text), 'Category']
        if len(row) == 0:
            return None
        return row.iloc[0]

    def category_to_text(self, category: str) -> Optional[str]:
        """
        :param category: Iconclass category
        :return: German text that is associated with the input category.
        """
        return self.__df.loc[(self.__df['Category'] == category), 'Text'].iloc[0]

    @staticmethod
    def get_category_at_level_or_higher(category: str, level: int) -> str:
        """
        :param category: Category of the iconclass tree
        :param level: Level which should ideally used to describe an artwork. (level >= 1)
        :return: Iconclass category at depth specified by parameter "level".
        If the category is not as deep down in the tree as specified by "level", then the full category is returned.
        """
        assert level >= 1
        available: List[str] = iconclass.get_parts(category)
        if len(available) <= level:
            return available[len(available) - 1]
        else:
            return available[level - 1]

    def get_all_missing_tags(self, all_tags: Set[str]) -> Set[str]:
        """
        :param all_tags: All text tags that are used in the dataset
        :return: Set with all tags that could not be found in the iconclass library
        """
        missing_tags: Set[str] = set()
        for c in all_tags:
            if self.text_to_category(c) is None:
                missing_tags.add(c)
        return missing_tags
