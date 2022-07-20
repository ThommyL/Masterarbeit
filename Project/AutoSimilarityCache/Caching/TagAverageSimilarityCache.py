"""
TagAverageSimilarityCache
"""

import functools
import os
import random
import threading
from typing import Dict, Tuple, Set, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from Project.AutoSimilarityCache.Caching import TAG_AVERAGE_SIMILARITY_CACHES_PATH
from Project.AutoSimilarityCache.Caching.TagVectorCache import TagVectorCache
from Project.AutoSimilarityCache.Misc.Misc import filter_and_optimize_data_tuples, split_list_into_equal_parts
from Project.AutoSimilarityCache.TagOriginUtils.OriginProcessing import validate_combined_origin_name, \
    combined_origin_to_origin_tuple, generate_df_name, get_max_amount_of_tags_of_loaded_origins
from Project.AutoSimilarityCacheConfiguration.ConfigurationMethods import get_similarity, reduce_to_one_of_type
from Project.AutoSimilarityCacheConfiguration.DataAccess import DataAccess
from Project.Utils.ConfigurationUtils.ConfigLoader import ConfigLoader
from Project.Utils.Misc.OriginContainer import OriginContainer
from Project.Utils.Misc.ProcessAndThreadHandlers import ProcessHandler, ProcessOrThreadHandler
from Project.Utils.Misc.Singleton import Singleton


@Singleton
class TagAverageSimilarityCache:
    """
    Caches the average similarities that a tag has to all identifiers.

    Results are stored in np.float32, since that is not just precise enough, but also the similarity calculation itself
    is imprecise, so the decimal places that are lost were probably wrong anyway.
    """

    def __init__(self):
        self.__dir_path: str = TAG_AVERAGE_SIMILARITY_CACHES_PATH
        self.__tag_similarity_path_dict: Optional[Dict[str, str]] = dict()
        self.__tag_similarity_dataframe_dict: Dict[str, Optional[pd.DataFrame]] = dict()
        self.__identifier_similarity_path_dict: Dict[str, str] = dict()
        self.__identifier_similarity_dataframe_dict: Dict[str, Optional[pd.DataFrame]] = dict()
        self.__lock: threading.Lock = threading.Lock()
        cl: ConfigLoader = ConfigLoader.instance
        self.__combined_origin_tuple: Tuple[str, ...] = cl.combined_origins_to_consider

        ph: ProcessHandler = ProcessHandler.instance

        for s in self.__combined_origin_tuple:
            self.__tag_similarity_path_dict[s] = os.path.join(
                self.__dir_path, 'tag_average_similarity_cache_' + s + '.pkl')
            self.__tag_similarity_dataframe_dict[s] = None

            self.__identifier_similarity_path_dict[s] = os.path.join(
                self.__dir_path, 'identifier_average_similarity_cache_' + s + '.pkl')
            self.__identifier_similarity_dataframe_dict[s] = None

            if not os.path.exists(self.__tag_similarity_path_dict[s]):
                while True:
                    if not ph.acquire_data_generation_lock():
                        # If this was generated in the meantime
                        if os.path.exists(self.__tag_similarity_path_dict[s]):
                            break
                        random.seed(0)
                        self.__generate_similarities_for_all_tags(combined_origin=s)
                        self.__tag_similarity_dataframe_dict[s].copy().to_pickle(self.__tag_similarity_path_dict[s])
                        break
                ph.release_data_generation_lock()

            if not os.path.exists(self.__identifier_similarity_path_dict[s]):
                while True:
                    if not ph.acquire_data_generation_lock():
                        # If this was generated in the meantime
                        if os.path.exists(self.__identifier_similarity_path_dict[s]):
                            break

                        self.__generate_average_similarities_for_all_identifiers(combined_origin=s)
                        self.__identifier_similarity_dataframe_dict[s].copy().to_pickle(
                            self.__identifier_similarity_path_dict[s])
                        break
                ph.release_data_generation_lock()

    @functools.lru_cache(maxsize=get_max_amount_of_tags_of_loaded_origins())
    def get_average_similarity_of_tag_tuple(
            self, tag_tuple: Tuple[any, str], origin_container: OriginContainer) -> np.float32:
        """
        :param tag_tuple: The tag tuple for which to check the average similarity
        :param origin_container: OriginContainer specifying which origins to consider
        :return: The average similarity a tag has to all other tags
        """
        df_name: str
        df_name = generate_df_name(origin_container=origin_container, is_rank=False)
        df: pd.DataFrame = self.__get_tag_cache_dataframe(combined_origin=df_name)
        result: np.float32 = df[df['TagTuple'] == tag_tuple]['AverageSimilarity'].iloc[0]
        assert result != np.float32(-99), 'Entry appears to not have been generated.'
        return result

    def __get_tag_cache_dataframe(self, combined_origin: str) -> pd.DataFrame:
        """
        :param combined_origin: Origin(s) for which to return the dataframe
        :return: Dataframe according to specified origins
        """
        if self.__tag_similarity_dataframe_dict[combined_origin] is not None:
            return self.__tag_similarity_dataframe_dict[combined_origin]

        self.__tag_similarity_dataframe_dict[combined_origin] = pd.read_pickle(
            self.__tag_similarity_path_dict[combined_origin])
        return self.__tag_similarity_dataframe_dict[combined_origin]

    def __get_identifier_cache_dataframe(self, combined_origin: str) -> pd.DataFrame:
        """
        :param combined_origin: Origin(s) for which to return the dataframe
        :return: Dataframe according to specified origins
        """
        if self.__identifier_similarity_dataframe_dict[combined_origin] is not None:
            return self.__identifier_similarity_dataframe_dict[combined_origin]

        self.__identifier_similarity_dataframe_dict[combined_origin] = pd.read_pickle(
            self.__identifier_similarity_path_dict[combined_origin])
        return self.__identifier_similarity_dataframe_dict[combined_origin]

    @functools.lru_cache()
    def get_average_similarity_for_identifier(self, identifier: str, combined_origin: str) -> float:
        """
        :param identifier: Identifier
        :param combined_origin: Combined origin of tags that should be considered for the calculation
        :return: Average similarity of the tags of the given identifier according to the input
        """
        validate_combined_origin_name(combined_origin=combined_origin)
        da: DataAccess = DataAccess.instance
        da.is_valid_identifier(identifier=identifier)
        df = self.__get_identifier_cache_dataframe(combined_origin=combined_origin)
        return df[df['Identifier'] == identifier]['AverageSimilarity'].iloc[0]

    def __generate_average_similarities_for_all_identifiers(self, combined_origin: str) -> None:
        """
        :param combined_origin: Combined Origin for which to generate the cache
        :return: None
        """
        da: DataAccess = DataAccess.instance
        self.__identifier_similarity_dataframe_dict[combined_origin] = pd.DataFrame(
            [], columns=['Identifier', 'AverageSimilarity']).astype({'Identifier': str, 'AverageSimilarity': float})
        ph: ProcessHandler = ProcessHandler.instance
        th: ProcessOrThreadHandler = ph.get_thread_handler()
        th.exec_in_pool(function=self.__thread_handled_generate_identifier_similarities,
                        maximum=len(da.get_ids()), args=(tuple(da.get_ids()), combined_origin))
        self.__identifier_similarity_dataframe_dict[combined_origin].copy().to_pickle(
            self.__identifier_similarity_path_dict[combined_origin])

    def __thread_handled_generate_identifier_similarities(self, identifiers: Set[str], combined_origin: str,
                                                          thread_handler_automatic_parameter=None) -> None:
        """A callback method that is meant to be called by the ThreadHandler class only. This method exists so that a
        wrapping method from the ThreadHandler class can assign threads to this method.
        :param identifiers: Identifiers for which to generate the results
        :param combined_origin: Combined Origin for which to generate the results
        :param thread_handler_automatic_parameter: This parameter will be set automatically by the ThreadHandler class
        :return: None
        """
        if thread_handler_automatic_parameter > 0:
            tuple_of_identifiers: Tuple[Tuple[Tuple[any, str], ...], ...] = split_list_into_equal_parts(
                tuple(identifiers), thread_handler_automatic_parameter)
            threads = set(threading.Thread(target=self.__generate_dataframe_for_identifiers,
                                           args=(c, combined_origin)) for c in tuple_of_identifiers)

            ph: ProcessHandler = ProcessHandler.instance
            ph.execute_threads(threads=threads)
        else:
            self.__generate_dataframe_for_identifiers(identifiers=identifiers, combined_origin=combined_origin)

    def __generate_dataframe_for_identifiers(self, identifiers: Set[str], combined_origin: str) -> None:
        """
        :param identifiers: Identifiers for which to generate the results
        :param combined_origin: Combined Origin for which to generate the results
        :return: None
        """
        result_dict: Dict[str, float] = dict()

        origin_container: OriginContainer = OriginContainer(
            combined_origin_to_origin_tuple(combined_origin=combined_origin))
        for i in tqdm(identifiers, desc='Generating average similarities for identifiers of combined origin '
                                        f'{combined_origin}'):
            similarities: List[float] = []
            for tag_tuple in filter_and_optimize_data_tuples(identifier=i, origin_container=origin_container):
                similarities.append(self.get_average_similarity_of_tag_tuple(
                    tag_tuple=tag_tuple, origin_container=origin_container))
            if len(similarities) == 0:
                result_dict[i] = 0
            else:
                result_dict[i] = sum(similarities) / len(similarities)
        self.__lock.acquire(blocking=True)
        self.__identifier_similarity_dataframe_dict[combined_origin] = self.__identifier_similarity_dataframe_dict[
            combined_origin].append(
            pd.DataFrame(result_dict.items(), columns=['Identifier', 'AverageSimilarity']), ignore_index=True)
        self.__lock.release()

    def __generate_similarities_for_all_tags(self, combined_origin: str) -> None:
        """
        Generates all dataframe entries in the dataframe corresponding to the specified combined origin
        :param combined_origin: Combined origin corresponding to dataframe for which to generate entries
        :return: None
        """
        validate_combined_origin_name(combined_origin=combined_origin)
        all_tag_tuples: Set[Tuple[any, str]] = set()
        compare_against: List[Tuple[any, str]] = []

        da: DataAccess = DataAccess.instance
        self.__tag_similarity_dataframe_dict[combined_origin] = pd.DataFrame(
            [], columns=['TagTuple', 'AverageSimilarity'])

        origin_container: OriginContainer = OriginContainer(combined_origin_to_origin_tuple(combined_origin))
        all_tag_tuples_with_weights: List[Tuple[any, str, float]] = []
        for i in tqdm(da.get_ids(), desc='Collecting all tag tuples and weights'):
            for tt in filter_and_optimize_data_tuples(identifier=i, origin_container=origin_container):
                all_tag_tuples_with_weights.append(
                    (tt[0], tt[1], da.get_weight_for_identifier_tag_tuple(identifier=i, tag_origin_tuple=tt)))
        random.shuffle(all_tag_tuples_with_weights)

        cl: ConfigLoader = ConfigLoader.instance
        max_size: int = cl.weighted_average_similarity_random_sample_size(combined_origin=combined_origin)

        for ttw in tqdm(all_tag_tuples_with_weights, desc="Performing random selection",
                        total=min(len(all_tag_tuples_with_weights), max_size)):
            if len(compare_against) > max_size:
                break
            if random.random() <= ttw[2]:
                compare_against.append((ttw[0], ttw[1]))

        for identifier in tqdm(da.get_ids(), desc='Collecting all tags'):
            current_tag_tuples = da.get_tag_tuples_from_identifier(identifier=identifier,
                                                                   origin_container=origin_container)
            for tt in current_tag_tuples:
                all_tag_tuples.add(tt)

        same_as_dict: Dict[Tuple[any, str], Tuple[any, str]]
        all_tag_tuples, same_as_dict = reduce_to_one_of_type(tuple(all_tag_tuples))

        ph: ProcessHandler = ProcessHandler.instance
        th: ProcessOrThreadHandler = ph.get_thread_handler()

        th.exec_in_pool(function=self.__thread_handled_generate_tag_similarities,
                        maximum=len(all_tag_tuples), args=(set(all_tag_tuples), tuple(compare_against), same_as_dict,
                                                           combined_origin))

        self.__tag_similarity_dataframe_dict[combined_origin].reset_index()
        self.__tag_similarity_dataframe_dict[combined_origin].copy().to_pickle(
            self.__tag_similarity_path_dict[combined_origin])

    def __thread_handled_generate_tag_similarities(
            self, all_tag_tuples: Set[Tuple[any, str]], compare_against: Tuple[Tuple[any, str], ...],
            same_as_dict: Dict[Tuple[any, str], Tuple[any, str]], combined_origin: str,
            thread_handler_automatic_parameter=None) -> None:
        """
        A callback method that is meant to be called by the ThreadHandler class only. This method exists so that a
        wrapping method from the ThreadHandler class can assign threads to this method.
        :param all_tag_tuples: Tag tuples for which to calculate entries / Tag tuples which are compared against to
        find the average.
        :param compare_against: Tag tuples which are compared against to find the approximate average.
        :param same_as_dict: Dictionary with tag tuples as keys for which no entries should be calculated, but instead
        the values should be copied from tag tuples given as values.
        :param combined_origin: Combined origin corresponding to dataframe for which to generate entries
        :param thread_handler_automatic_parameter: This parameter will be set automatically by the ThreadHandler class
        :return: None
        """
        if thread_handler_automatic_parameter > 0:
            tuple_of_tags: Tuple[Tuple[Tuple[any, str], ...], ...] = split_list_into_equal_parts(
                tuple(all_tag_tuples), thread_handler_automatic_parameter)
            threads: Set[threading.Thread] = set(
                threading.Thread(target=self.__generate_dataframe_for_tags, args=(
                    c, same_as_dict, compare_against, combined_origin)) for c in tuple_of_tags)
            ph: ProcessHandler = ProcessHandler.instance
            ph.execute_threads(threads=threads)
        else:
            self.__generate_dataframe_for_tags(
                tag_tuples_to_process=all_tag_tuples, same_as_dict=same_as_dict,
                compare_against=compare_against, combined_origin=combined_origin)

    def __generate_dataframe_for_tags(
            self, tag_tuples_to_process: Set[Tuple[any, str]],
            same_as_dict: Dict[Tuple[any, str], Tuple[any, str]], compare_against: Tuple[Tuple[any, str], ...],
            combined_origin: str) -> None:
        """
        Fills tags of the dataframe belonging to the specified origin with the calculated results. Note, this method
        does not save the dataframe.
        :param tag_tuples_to_process: Tags for which to generate the results
        :param compare_against: Tag tuples which are compared against to find the approximate average.
        :param same_as_dict: Dictionary with tag tuples as keys for which no entries should be calculated, but instead
        the values should be copied from tag tuples given as values.
        :param combined_origin: Origin belonging to the dataframe for which to generate the results
        :return: None
        """
        origin_container: OriginContainer = OriginContainer(
            combined_origin_to_origin_tuple(combined_origin=combined_origin))

        cl: ConfigLoader = ConfigLoader.instance
        assert not origin_container.origin_is_true('Des') or cl.allow_description_tags, \
            'Description tags are forbidden by the config file parameter \'allow-description-tags\'.'

        tvc: TagVectorCache = TagVectorCache.instance
        result_dict: Dict[Tuple[any, str], np.float32] = dict()
        for ind, tag_tuple in enumerate(
                tqdm(tag_tuples_to_process,
                     desc=f'Generating average similarities for tags for origin \'{combined_origin}\'')):
            all_sims = []
            for tt in compare_against:
                all_sims.append(
                    get_similarity(
                        vector_origin_tuple_1=(tvc.get_vector_for_tag_tuple(tag_origin_tuple=tag_tuple), tag_tuple[1]),
                        vector_origin_tuple_2=(tvc.get_vector_for_tag_tuple(tag_origin_tuple=tt), tt[1]))
                )
            result = 0
            if len(all_sims) > 0:
                result = sum(all_sims) / len(all_sims)

            result_dict[tag_tuple] = np.float32(result)

        for k in same_as_dict.keys():
            if same_as_dict[k] in result_dict:
                result_dict[k] = result_dict[same_as_dict[k]]

        self.__lock.acquire(blocking=True)
        self.__tag_similarity_dataframe_dict[combined_origin] = self.__tag_similarity_dataframe_dict[
            combined_origin].append(
            pd.DataFrame(result_dict.items(), columns=['TagTuple', 'AverageSimilarity']), ignore_index=True)
        self.__lock.release()
