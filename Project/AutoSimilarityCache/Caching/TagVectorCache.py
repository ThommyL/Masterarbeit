"""
TagVectorCache
"""

import functools
import os
from typing import Set, Dict, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from Project.AutoSimilarityCache.Caching import TAG_VECTOR_CACHE_PATH
from Project.AutoSimilarityCache.TagOriginUtils.OriginProcessing import get_max_amount_of_tags_of_loaded_origins
from Project.AutoSimilarityCacheConfiguration.ConfigurationMethods import get_vector, reduce_to_one_of_type
from Project.AutoSimilarityCacheConfiguration.DataAccess import DataAccess
from Project.Utils.ConfigurationUtils.ConfigLoader import ConfigLoader
from Project.Utils.Misc.ProcessAndThreadHandlers import ProcessHandler
from Project.Utils.Misc.Singleton import Singleton


@Singleton
class TagVectorCache:
    """
    Caches the spacy docs of all tags in the cleaned data into a pandas DataFrame. This speeds up things significantly.
    Note: Not multi-threaded since this cache does not take very long to generate and this way the get_vector method
    of ConfigurationMethods.py does not need to be threadsafe.
    """

    def __init__(self):
        self.__dir_path: str = TAG_VECTOR_CACHE_PATH
        da: DataAccess = DataAccess.instance
        pickle_path = os.path.join(TAG_VECTOR_CACHE_PATH, 'Tags_Vector_Dict.pkl')
        self.__df = None

        ph: ProcessHandler = ProcessHandler.instance

        if not os.path.exists(pickle_path):
            while True:
                if not ph.acquire_data_generation_lock():
                    # If this was generated in the meantime
                    if os.path.exists(pickle_path):
                        self.__df = pd.read_pickle(pickle_path)
                        break
                    all_tag_tuples: Set[Tuple[any, str]] = set()
                    cl: ConfigLoader = ConfigLoader.instance
                    for i in tqdm(da.get_ids(), desc="Collecting all tags"):
                        for tag_tuple in da.get_tag_tuples_from_identifier(
                                i, cl.get_origin_container_with_possible_origins()):
                            all_tag_tuples.add(tag_tuple)

                    same_as_dict: Dict[Tuple[any, str], Tuple[any, str]]
                    all_tag_tuples, same_as_dict = reduce_to_one_of_type(tuple(all_tag_tuples))

                    all_rows: Dict[Tuple[any, str], np.ndarray] = dict()
                    for i, tag_tuple in enumerate(tqdm(tuple(all_tag_tuples), desc="Initializing tag vector cache")):
                        all_rows[tag_tuple] = get_vector(tag_origin_tuple=tag_tuple)
                    for k in same_as_dict.keys():
                        all_rows[k] = all_rows[same_as_dict[k]]

                    df: pd.DataFrame = pd.DataFrame(
                        all_rows.items(), columns=('TagTuple', 'Vector')).drop_duplicates(subset='TagTuple')
                    df.copy().to_pickle(pickle_path)
                    self.__df = df
                    break
            ph.release_data_generation_lock()
        else:
            self.__df = pd.read_pickle(pickle_path)

        # Keeping only relevant entries:
        cl: ConfigLoader = ConfigLoader.instance
        relevant_entries: Dict[Tuple[any, str], np.array] = dict()
        for _, row in self.__df.iterrows():
            current: Tuple[any, str] = row['TagTuple']
            if current[1] in cl.active_origins:
                relevant_entries[current] = row['Vector']
        self.__df = pd.DataFrame(relevant_entries.items(), columns=['TagTuple', 'Vector'])

    @functools.lru_cache(maxsize=get_max_amount_of_tags_of_loaded_origins())
    def get_vector_for_tag_tuple(self, tag_origin_tuple: Tuple[any, str]) -> np.array:
        """
        :param tag_origin_tuple: The tag for which to return the vector
        :return: The vector of the tag specified
        """
        return self.__df[self.__df['TagTuple'] == tag_origin_tuple]['Vector'].iloc[0]
