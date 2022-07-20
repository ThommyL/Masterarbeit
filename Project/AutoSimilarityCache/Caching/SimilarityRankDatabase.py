"""
SimilarityRankDatabase.py
"""

import itertools
import os
import pickle
import threading
import warnings
from typing import List, Dict, Tuple, Set, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from Project.AutoSimilarityCache.Caching import SIMILARITY_AND_RANK_CACHES_PATH, \
    SIMILARITY_AND_RANK_CACHES_GENERATED_IDENTIFIERS_PATH, ALL_VALUES_CHECKED, IDENTIFIER_COLUMN_NAME
from Project.AutoSimilarityCache.Caching.TagAverageSimilarityCache import TagAverageSimilarityCache
from Project.AutoSimilarityCache.Caching.TagVectorCache import TagVectorCache
from Project.AutoSimilarityCache.Misc.Misc import filter_and_optimize_data_tuples
from Project.AutoSimilarityCache.TagOriginUtils.OriginProcessing import generate_df_name, \
    origin_to_origin_container, logical_or_for_origin_containers, combined_origin_to_origin_container
from Project.AutoSimilarityCacheConfiguration.ConfigurationMethods import get_similarity
from Project.AutoSimilarityCacheConfiguration.DataAccess import DataAccess
from Project.Utils.ConfigurationUtils.ConfigLoader import ConfigLoader
from Project.Utils.Misc.OriginContainer import OriginContainer
from Project.Utils.Misc.ProcessAndThreadHandlers import ProcessOrThreadHandler, ProcessHandler
from Project.Utils.Misc.Singleton import Singleton


@Singleton
class SimilarityRankDatabase:
    """
    Singleton Pattern that provides a dataframe of len(dataframe)-nearest neighbors. Dataframe needs to be initialized
    with generate_similarity_dataframe() method in Misc.py first. Here a csv file is chosen for storage instead of
    a pickle file since it is more efficient to append to a csv file line by line.
    Note that if this class is used in test mode (see config-file-parameter data-generation-mode), then the database
    must be saved manually by calling the method "save_generated".
    """

    def __init__(self):
        self.__dir_path: str = SIMILARITY_AND_RANK_CACHES_PATH
        self.__save_lock: threading.Lock = threading.Lock()
        self.__dataframe_path_dictionary: Dict[str, str] = dict()

        cl: ConfigLoader = ConfigLoader.instance

        self.__dataframes: Tuple[str, ...] = cl.currently_possible_combined_origins
        self.__rank_dataframes: Tuple[str, ...] = tuple('rank_' + d for d in self.__dataframes)
        self.__correction_dataframes: Tuple[str, ...] = tuple('correction_' + d for d in self.__dataframes)

        for df in self.__dataframes + self.__rank_dataframes + self.__correction_dataframes:
            self.__dataframe_path_dictionary[df] = os.path.join(self.__dir_path, df + '.pkl')
        self.__dataframe_dict = dict()

        self.__locks: Optional[Dict[str, threading.Lock]] = None

        if cl.data_generation_mode_is_test:
            self.__locks = dict()
            for s in cl.combined_origins_to_consider:
                self.__locks[s] = dict()

        self.__thread_lock: Dict[str, Dict[str, int]] = dict()
        self.__dataframe_lock: Dict[str, threading.Lock()] = dict()
        self.__already_generated_dict_lock: Dict[str, threading.Lock] = dict()
        if cl.database_generation_in_progress:
            for s in cl.combined_origins_to_consider:
                self.__thread_lock[s] = dict()
                self.__dataframe_lock[s] = threading.Lock()
                self.__dataframe_lock['rank_' + s] = threading.Lock()
                self.__already_generated_dict_lock[s] = threading.Lock()

        for s in cl.combined_origins_to_consider:
            self.__dataframe_dict[s] = self.__get_dataframe(name=s)
            self.__dataframe_dict['rank_' + s] = self.__get_dataframe(name='rank_' + s)

        self.__active_dict_names: Tuple[str, ...] = cl.currently_possible_combined_origins

        self.__already_generated_dict = None
        if cl.database_generation_in_progress:
            self.__already_generated_dict: Dict[str, List[str]] = dict()

        cl: ConfigLoader = ConfigLoader.instance
        self.__all_generated: bool
        if not os.path.exists(ALL_VALUES_CHECKED + '.pkl'):
            with open(ALL_VALUES_CHECKED + '.pkl', 'wb+') as f:
                all_values_checked: Dict[str, bool] = dict()
                for n in cl.possible_dictionary_names:
                    all_values_checked[n] = False
                pickle.dump(all_values_checked, f)
            self.__all_generated = False
        else:
            with open(ALL_VALUES_CHECKED + '.pkl', 'rb') as f:
                all_values_checked = pickle.load(f)

            self.__all_generated = True
            for s in cl.combined_origins_to_consider:
                if not all_values_checked[s]:
                    self.__all_generated = False
                    break

        assert self.__all_generated or cl.data_generation_mode_is_test, 'In production mode all values must be ' \
                                                                        'generated.'

    def generate_similarity_and_rank_caches(self) -> None:
        """
        Generates the similarity and rank caches of the active combined origins in an efficient manner.
        :return: None
        """
        cl: ConfigLoader = ConfigLoader.instance
        co_length_dict: Dict[str, int] = dict()
        for co in cl.combined_origins_to_consider:
            co_length_dict[co] = len(co.split('&'))
        to_remove: Set[str] = set()
        for i in range(min(co_length_dict.values()), max(co_length_dict.values())):
            candidates = tuple(k for k in co_length_dict.keys() if co_length_dict[k] == i)
            for c in candidates:
                c_parts = c.split('&')
                contained: bool = False
                for j in range(i + 1, max(co_length_dict.values()) + 1):
                    compare_to = tuple(k for k in co_length_dict.keys() if co_length_dict[k] == j)
                    for ct in compare_to:
                        ct_parts = ct.split('&')
                        match: bool = True
                        first_match_found: bool = False
                        offset: int = 0
                        overlap: int = 0

                        for k in range(len(ct_parts)):
                            if c_parts[k - offset] == ct_parts[k]:
                                first_match_found = True

                            if first_match_found:
                                if c_parts[k - offset] == ct_parts[k]:
                                    overlap += 1
                                    if overlap == len(c_parts):
                                        break
                                else:
                                    match = False
                                    break
                            else:
                                offset += 1
                        match = match and first_match_found
                        if match:
                            contained = True
                            break
                    if contained:
                        to_remove.add(c)
                        break
        to_generate: Tuple[str, ...] = tuple(co for co in cl.combined_origins_to_consider if co not in to_remove)

        da: DataAccess = DataAccess.instance
        for ind, i in tqdm(enumerate(da.get_ids()), desc='Generating Similarity Cache'):
            for g in to_generate:
                _ = self.get_similarities_for_id(identifier=i, origin_container=combined_origin_to_origin_container(g))
            if ind % 500 == 0:
                self.save_generated()
        self.save_generated()

        for i in tqdm(da.get_ids(), desc='Generating Rank Cache'):
            for co in cl.combined_origins_to_consider:
                _ = self.get_ranks_for_id(identifier=i, origin_container=combined_origin_to_origin_container(co))
        self.save_generated()

        with open(ALL_VALUES_CHECKED + '.pkl', 'rb') as f:
            all_values_checked = pickle.load(f)

        with open(ALL_VALUES_CHECKED + '.pkl', 'wb') as f:
            for g in to_generate:
                all_values_checked[g] = True
            pickle.dump(all_values_checked, f)

        self.save_generated()

    def __save_already_generated_dict(self, combined_origin: str) -> None:
        """
        Writes the dictionary with already generated identifiers of all combined origins to disk
        :return: None
        """
        if combined_origin not in self.__already_generated_dict.keys():
            return
        with open(SIMILARITY_AND_RANK_CACHES_GENERATED_IDENTIFIERS_PATH + '_' + combined_origin + '.pkl', 'wb+') as f:
            pickle.dump(self.__get_already_generated_ids(combined_origin), f)

    def __get_already_generated_ids(self, combined_origin: str) -> List[str]:
        """
        :param combined_origin: Combined origin to which the return value should correspond to
        :return: A list containing all identifiers for which similarities have already been generated
        """
        if combined_origin not in self.__already_generated_dict.keys():
            if os.path.exists(
                    path := SIMILARITY_AND_RANK_CACHES_GENERATED_IDENTIFIERS_PATH + '_' + combined_origin + '.pkl'):
                with open(path, 'rb') as f:
                    self.__already_generated_dict[combined_origin] = pickle.load(f)
            else:
                self.__already_generated_dict[combined_origin] = []
        return self.__already_generated_dict[combined_origin]

    def __append_to_already_generated_dict(self, identifier: str, combined_origin: str) -> None:
        """
        Appends the list of already generated identifiers corresponding to the specified combined origin in a thread
        safe manner
        :param combined_origin: Combined origin that corresponds to the list that should be appended
        :return: None
        """
        cl: ConfigLoader = ConfigLoader.instance
        assert cl.database_generation_in_progress
        self.__already_generated_dict_lock[combined_origin].acquire(blocking=True, timeout=-1)
        self.__get_already_generated_ids(combined_origin).append(identifier)
        self.__already_generated_dict_lock[combined_origin].release()

    def get_already_generated_entries(self, combined_origin: str) -> Tuple[str, ...]:
        """
        Note: Intended for use in development and testing
        :param combined_origin: Combined Origin for which to check
        :return: Tuple containing all identifiers for which the cache of the given combined origin has already been
        generated
        """
        return tuple(self.__get_already_generated_ids(combined_origin))

    def __get_empty_dataframe(self, ranked: bool) -> pd.DataFrame:
        """
        Since it takes very long to generate an empty dataset, this method return a template emtpy dataset. If necessary
        this dataset will be newly generated.
        :param ranked: If True an empty dataset for a rank dataset will be returned, otherwise an empty dataset for
        a similarity dataset will be returned.
        :return: The requested empty dataset
        """
        file_name: str = 'empty_similarity_dataframe'
        if ranked:
            file_name += '_rank'
        file_name += '.pkl'
        empty_df_path: str = os.path.join(self.__dir_path, file_name)
        if not os.path.exists(empty_df_path):
            df: pd.DataFrame = self.__generate_empty_dataframe(ranked=ranked)
            df.copy().to_pickle(empty_df_path)
        return pd.read_pickle(empty_df_path)

    def __get_dataframe(self, name: str) -> pd.DataFrame:
        """
        Method that loads a dataframe if not yet loaded and returns it. If this dataframe file does not yet exist, it
        will be generated first. This can take over an hour.
        :param name: The name of the dataframe which to load
        :return: The requested pandas dataframe
        """
        assert name in self.__dataframes or name in self.__rank_dataframes or name, 'Name not valid'
        if name in self.__dataframe_dict.keys():
            return self.__dataframe_dict[name]
        cur_path: str = self.__dataframe_path_dictionary[name]

        if not os.path.exists(cur_path):
            self.__get_empty_dataframe(ranked=name.startswith('rank_')).copy().to_pickle(cur_path)
        self.__dataframe_dict[name] = pd.read_pickle(cur_path)
        return self.__dataframe_dict[name]

    @staticmethod
    def __get_type_dict(ranked: bool) -> Dict[str, type]:
        """
        :param ranked: If True the types that are returned correspond to a rank dataframe, otherwise they correspond
        to a similarity dataframe
        :return: The column types for the requested type of dataframe
        """
        da: DataAccess = DataAccess.instance
        if ranked:
            columns: Tuple[str, ...] = tuple([IDENTIFIER_COLUMN_NAME] + [str(i) for i in range(1, len(da.get_ids()))])
        else:
            columns: Tuple[str, ...] = tuple([IDENTIFIER_COLUMN_NAME] + [i for i in da.get_ids()])

        type_dict: Dict[str, type] = dict()
        for c in columns:
            if ranked:
                type_dict[c] = str
            else:
                if c == IDENTIFIER_COLUMN_NAME:
                    type_dict[c] = str
                else:
                    type_dict[c] = np.float32
        return type_dict

    def __generate_empty_dataframe(self, ranked: bool) -> pd.DataFrame:
        """
        :param ranked: If true returns an empty dataframe for ranks else return an empty dataframe for similarities
        :return: The Dataframe with the additional rows
        """
        da: DataAccess = DataAccess.instance
        identifiers: Tuple[str, ...] = da.get_ids()
        type_dict: Dict[str, type] = self.__get_type_dict(ranked=ranked)
        all_rows: List[Dict] = []
        if ranked:
            columns: Tuple = tuple([IDENTIFIER_COLUMN_NAME] + [str(i) for i in range(1, len(da.get_ids()))])

            for identifier in tqdm(identifiers, desc=f'Adding identifiers to rank dataframe'):
                current = dict()
                for c in columns:
                    if c == IDENTIFIER_COLUMN_NAME:
                        current[c] = identifier
                    else:
                        current[c] = ''
                all_rows.append(current)
        else:
            columns: Tuple = tuple([IDENTIFIER_COLUMN_NAME] + [i for i in da.get_ids()])

            for i in tqdm(identifiers, desc=f'Adding identifiers to similarity dataframe'):
                current = dict()
                for c in columns:
                    if c == IDENTIFIER_COLUMN_NAME:
                        current[c] = i
                    else:
                        current[c] = np.nan
                all_rows.append(current)
        df = pd.DataFrame(all_rows, columns=columns)
        df = df.astype(type_dict)
        return df

    @staticmethod
    def __row_to_dict(row: pd.DataFrame, identifier: str, ranked: bool) -> Dict[str, any]:
        """
        A method that converts a given row of a dataframe of this class to a dictionary
        :param row: The row
        :param identifier: The identifier to which this row belongs. It is filtered out in the result.
        :return: The row as dictionary
        """
        res: Dict = dict()
        if ranked:
            da: DataAccess = DataAccess.instance
            all_ids: Tuple[str, ...] = da.get_ids()
            correction: int = 0
            for i in range(1, len(all_ids)):
                if all_ids[i] == identifier:
                    correction = 1
                    continue
                res[row[i].iloc[0]] = i - correction
        else:
            da: DataAccess = DataAccess.instance
            for i in da.get_ids():
                if i == identifier:
                    continue
                res[i] = row[i].iloc[0]
        return res

    def __acquire_lock(self, identifier, combined_origin: str) -> Tuple[bool, bool]:
        """
        :param identifier: Parameter for which to acquire the lock
        :param combined_origin: The combined origin for which to acquire the lock for
        :return: Tuple: had_to_wait_for_lock, release_lock_afterwards
        """
        cl: ConfigLoader = ConfigLoader.instance
        assert cl.data_generation_mode_is_test, 'These locks are only necessary in test mode'
        if identifier in self.__thread_lock[combined_origin].keys() and \
                self.__thread_lock[combined_origin][identifier] == threading.get_ident():
            return False, False
        if identifier not in self.__locks[combined_origin]:
            self.__locks[combined_origin][identifier] = threading.Lock()
        was_locked = not self.__locks[combined_origin][identifier].acquire(blocking=False)
        if was_locked:
            self.__locks[combined_origin][identifier].acquire(blocking=True, timeout=-1)
        self.__thread_lock[combined_origin][identifier] = threading.get_ident()
        return was_locked, True

    def __release_lock(self, identifier: str, combined_origin: str) -> None:
        """
        :param identifier: Identifier for which to release the lock
        :param combined_origin: The combined origin for which to release the lock
        :return: None
        """
        cl: ConfigLoader = ConfigLoader.instance
        assert cl.data_generation_mode_is_test, 'These locks are only necessary in test mode'
        del self.__thread_lock[combined_origin][identifier]
        self.__locks[combined_origin][identifier].release()

    def get_ranks_for_id(self, identifier: str, origin_container: OriginContainer) -> Dict[str, int]:
        """
        :param identifier: The sample for which to query
        :param origin_container: The origins of the tags that should be considered
        :return: A dictionary containing identifiers as keys and the respective ranks as value
        """
        cl: ConfigLoader = ConfigLoader.instance
        assert cl.is_valid_identifier(identifier), 'Could not find identifier in cleaned data'
        """
        Note: Threadsafety could be guaranteed in the same manner as in get_similarities_for_id, but since the method
        does not nearly take as long to execute as get_similarities_for_id, the damage is minimal if the result is
        calculated twice and this way the processing time of acquiring and releasing locks is avoided.
        """
        rank_df_name: str
        rank_df_name = generate_df_name(origin_container=origin_container, is_rank=True)
        rank_df: pd.DataFrame = self.__get_dataframe(rank_df_name)
        cl: ConfigLoader = ConfigLoader.instance
        ph: ProcessHandler = ProcessHandler.instance

        if not self.__all_generated:
            self.__dataframe_lock[rank_df_name].acquire(blocking=True, timeout=-1)
        row = rank_df[rank_df[IDENTIFIER_COLUMN_NAME] == identifier]
        if not self.__all_generated:
            self.__dataframe_lock[rank_df_name].release()

        if row.iloc[0].iloc[1] != '':
            if cl.similarity_rank_database_debug_output_enabled:
                print(
                    f'{ph.get_thread_name_and_register()}: I am fetching ranks for identifier \'{identifier}\' '
                    f'from cache for origins: {origin_container.to_origin_tuple()}.')
            return self.__row_to_dict(row=row, identifier=identifier, ranked=True)
        test_mode: bool = cl.data_generation_mode_is_test

        if cl.similarity_rank_database_debug_output_enabled:
            print(f'{ph.get_thread_name_and_register()}: I am generating ranks for: {identifier} '
                  f'with origin(s): {origin_container.to_origin_tuple()}')
        else:
            assert cl.dataframe_generation_in_progress, \
                f'Value was supposed to be generated, but could not be found: {identifier}'

        similarities: Dict[str, float] = self.get_similarities_for_id(identifier, origin_container=origin_container)
        ranks: Tuple[str, ...] = tuple(t[0] for t in sorted(similarities.items(), key=lambda x: x[1], reverse=True))

        self.__dataframe_lock[rank_df_name].acquire(blocking=True, timeout=-1)
        for rank, ident in enumerate(ranks):
            rank_df.loc[rank_df[IDENTIFIER_COLUMN_NAME] == identifier, rank + 1] = ident
        result = self.__row_to_dict(row=rank_df[rank_df[IDENTIFIER_COLUMN_NAME] == identifier], identifier=identifier,
                                    ranked=True)
        self.__dataframe_lock[rank_df_name].release()

        if test_mode:
            if cl.similarity_rank_database_debug_output_enabled:
                print(f'{ph.get_thread_name_and_register()}: I have finished generating ranks for identifier '
                      f'\'{identifier}\' with origin(s): {origin_container.to_origin_tuple()}')
        return result

    def get_similarities_for_id(self, identifier: str, origin_container: OriginContainer) -> Dict[str, float]:
        """
        :param identifier: The entity for which to query
        :param origin_container: Which origins from tags should be considered for the search ('Des', 'Title',
        'Exp', ...)
        :return: A tuple of Tuple[identifier, similarity]
        """
        da: DataAccess = DataAccess.instance
        assert da.is_valid_identifier(identifier), 'Could not find identifier in cleaned data.'

        df_name: str
        number_of_origins: int = origin_container.number_of_true_origins
        df_name = generate_df_name(origin_container=origin_container, is_rank=False)

        cur_df: pd.DataFrame = self.__get_dataframe(df_name)

        cl: ConfigLoader = ConfigLoader.instance
        ph: ProcessHandler = ProcessHandler.instance

        if self.__all_generated or identifier in self.__get_already_generated_ids(df_name):
            if cl.similarity_rank_database_debug_output_enabled:
                print(
                    f'{ph.get_thread_name_and_register()}: I am fetching similarities for identifier \'{identifier}\' '
                    f'from cache.')

            if not self.__all_generated:
                self.__dataframe_lock[df_name].acquire(blocking=True, timeout=-1)
            row = cur_df.loc[cur_df[IDENTIFIER_COLUMN_NAME] == identifier]
            if not self.__all_generated:
                self.__dataframe_lock[df_name].release()
            if cl.data_generation_mode_is_test:
                ph: ProcessHandler = ProcessHandler.instance
                th: ProcessOrThreadHandler = ph.get_thread_handler()
                th.register_idle('I am done fetching the similarities.')
            return self.__row_to_dict(row=row, identifier=identifier, ranked=False)

        release_lock_afterwards: bool = True
        if cl.database_generation_in_progress:
            if cl.similarity_rank_database_debug_output_enabled:
                print(f'{ph.get_thread_name_and_register()}: I am acquiring the lock for identifier \'{identifier}\' '
                      f'with origin(s): {origin_container.to_origin_tuple()}')
            had_to_wait, release_lock_afterwards = self.__acquire_lock(identifier, df_name)
            if had_to_wait:
                if cl.similarity_rank_database_debug_output_enabled:
                    print(f'{ph.get_thread_name_and_register()}: I had to wait for identifier \'{identifier}\' to be '
                          f'generated.')
                if identifier in self.__get_already_generated_ids(df_name):
                    self.__dataframe_lock[df_name].acquire(blocking=True, timeout=-1)
                    row = cur_df[cur_df[IDENTIFIER_COLUMN_NAME] == identifier]
                    self.__dataframe_lock[df_name].release()
                    return self.__row_to_dict(row=row,
                                              identifier=identifier, ranked=False)
                else:
                    raise Exception('Row for Identifier should have been generated, but was not.')
            if cl.similarity_rank_database_debug_output_enabled:
                print(f'{ph.get_thread_name_and_register()}: I am generating similarities for: \'{identifier}\''
                      f' with origin(s): {origin_container.to_origin_tuple()}')
        else:
            assert not cl.database_generation_in_progress, \
                f'Value was supposed to be generated, but could not be found: identifier = \'{identifier}\''

        da: DataAccess = DataAccess.instance

        own_docs: Tuple[Tuple[str, str], ...] = filter_and_optimize_data_tuples(
            identifier=identifier, origin_container=origin_container)

        # Generate all relevant rows for fast lookup
        if cl.database_generation_in_progress:
            ph: ProcessHandler = ProcessHandler.instance
            th: ProcessOrThreadHandler = ph.get_thread_handler()
            if number_of_origins > 1:
                generate_for: List[int] = [1]
                if number_of_origins > 2:
                    generate_for.append(2)
                for g in generate_for:
                    origin_tuples_on_current_level: List[Tuple[str, ...]] = []
                    for c in itertools.combinations(origin_container.to_origin_tuple(), g):
                        origin_tuples_on_current_level.append(c)
                    th.register_idle('I am waiting for similarities of other origins to be generated.')
                    self.__generate_similarities_for_multiple_origins(
                        identifier=identifier,
                        origin_containers=tuple(OriginContainer(o) for o in origin_tuples_on_current_level))
                    th.unregister_idle()

        # Load all relevant rows for fast lookup if necessary
        row_dict: Dict[str, Optional[pd.DataFrame]] = dict()
        if number_of_origins > 1:
            gen_for: List[int] = [1]
            if number_of_origins > 2:
                gen_for.append(2)
            for gen in gen_for:
                for comb in itertools.combinations(origin_container.to_origin_tuple(), gen):
                    cur_name: str
                    cur_name = generate_df_name(
                        OriginContainer(origins=comb), is_rank=False)
                    row_dict[cur_name] = None

        def get_row_dict(key: str) -> pd.DataFrame:
            """
            Loads dataframe rows as needed and returns them.
            Note: This method manipulates a variable from the outer scope.
            :param key: Name of the dataframe from which to fetch a value
            :return: The requested row as DataFrame.
            """

            # Note: this is a variable from the outer context
            if row_dict[key] is None:
                self.__dataframe_lock[df_name].acquire(blocking=True, timeout=-1)
                temp_df: pd.DataFrame = self.__get_dataframe(key)
                row_dict[key] = temp_df.loc[temp_df[IDENTIFIER_COLUMN_NAME] == identifier]
                self.__dataframe_lock[df_name].release()
            return row_dict[key]

        tvc: TagVectorCache = TagVectorCache.instance
        tasc: TagAverageSimilarityCache = TagAverageSimilarityCache.instance

        current_results: Dict[str, float] = dict()
        to_generate: Set[str] = set(da.get_ids()) - {identifier} - set(self.__get_already_generated_ids(df_name))

        for i in to_generate:
            sims: List[np.float32This] = []
            other_docs: Tuple[Tuple[str, str], ...] = filter_and_optimize_data_tuples(
                identifier=i, origin_container=origin_container)
            for own in own_docs:
                average_own: Optional[float] = None
                if number_of_origins <= 2:
                    average_own = tasc.get_average_similarity_of_tag_tuple(
                        tag_tuple=own, origin_container=origin_container)
                for other in other_docs:
                    if number_of_origins > 2 or number_of_origins > 1 and own[1] == other[1]:
                        fetch_from_df: str
                        fetch_from_df = generate_df_name(
                            origin_container=logical_or_for_origin_containers(
                                origin_to_origin_container(own[1]),
                                origin_to_origin_container(other[1])), is_rank=False)
                        sims.append(get_row_dict(fetch_from_df)[i].iloc[0])
                    else:
                        cur_sim: np.float32 = np.float32(get_similarity(
                            vector_origin_tuple_1=(tvc.get_vector_for_tag_tuple(tag_origin_tuple=own), own[1]),
                            vector_origin_tuple_2=(tvc.get_vector_for_tag_tuple(tag_origin_tuple=other), other[1])))
                        average_other: float = tasc.get_average_similarity_of_tag_tuple(
                            tag_tuple=other, origin_container=origin_container)

                        # Note: Although similarities are mostly between 0 and 1, the cosine distance goes from -1 to 1
                        sims.append((cur_sim - (average_own + average_other) / 2))
            result: np.float32 = np.float32(0)
            if len(sims) != 0:
                result = np.float32(sum(sims) / len(sims))

            # Theoretically it is possible that by subtracting the average similarities of tags the boundaries of
            # [-1, 1] are overstepped.
            if result < -1:
                warnings.warn(f'{identifier} generated a similarity of {result}. This was capped to -1.')
                result = np.float32(-1)
            elif result > 1:
                warnings.warn(f'{identifier} generated a similarity of {result}. This was capped to 1.')
                result = np.float32(1)

            current_results[i] = result

        self.__dataframe_lock[df_name].acquire(blocking=True, timeout=-1)
        for i in to_generate:
            cur_df.loc[cur_df[IDENTIFIER_COLUMN_NAME] == identifier, i] = current_results[i]
            cur_df.loc[cur_df[IDENTIFIER_COLUMN_NAME] == i, identifier] = current_results[i]
        result_dict: Dict[str, float] = self.__row_to_dict(row=cur_df.loc[cur_df[IDENTIFIER_COLUMN_NAME] == identifier],
                                                           identifier=identifier, ranked=False)
        self.__dataframe_lock[df_name].release()

        self.__append_to_already_generated_dict(identifier=identifier, combined_origin=df_name)

        if cl.database_generation_in_progress:
            ph: ProcessHandler = ProcessHandler.instance
            th: ProcessOrThreadHandler = ph.get_thread_handler()
            th.register_idle('I am done generating the similarities.')
            if cl.similarity_rank_database_debug_output_enabled:
                print(f'{ph.get_thread_name_and_register()}: I have finished generating similarities for: '
                      f'identifier \'{identifier}\' with origin(s): {origin_container.to_origin_tuple()}')
            if release_lock_afterwards:
                self.__release_lock(identifier, df_name)
        return result_dict

    def save_generated(self):
        """
        Saves the dataframes. Only to be used in test mode.
        :return: None
        """
        self.__save_lock.acquire(blocking=True)
        cl: ConfigLoader = ConfigLoader.instance
        assert cl.data_generation_mode_is_test, 'This method is only meant to be used in test mode'
        for o in cl.combined_origins_to_consider:
            self.__dataframe_dict[o].copy().to_pickle(self.__dataframe_path_dictionary[o])
            self.__save_already_generated_dict(o)
            self.__dataframe_dict['rank_' + o].copy().to_pickle(self.__dataframe_path_dictionary['rank_' + o])
        self.__save_lock.release()

    def identifiers_are_similar(self, id_1: str, id_2: str, origin_container: OriginContainer) -> float:
        """
        :param id_1: First identifier
        :param id_2: Second Identifier
        :param origin_container: Origins of tags which to consider for similarity calculation
        :return: True if cosine similarity of the two entities is positive, False otherwise
        """
        return self.get_similarities_for_id(identifier=id_1, origin_container=origin_container)[id_2] > 0

    def get_ranks_for_pair(self, id_1: str, id_2: str, origin_container: OriginContainer
                           ) -> Tuple[Dict[str, int], Dict[str, int]]:
        """
        :param id_1: The first identifier for which a rank dictionary should be returned
        :param id_2: The second identifier for which a rank dictionary should be returned
        :param origin_container: The origins of which tags should be considered
        :return: Tuple with 2 dictionaries corresponding to id_1 and id_2
        """
        ph: ProcessHandler = ProcessHandler.instance
        th: ProcessOrThreadHandler = ph.get_thread_handler()
        return th.exec_in_pool(function=self.__thread_handled_get_rank_dicts, maximum=2,
                               args=(id_1, id_2, origin_container))

    def __thread_handled_get_rank_dicts(self, id_1: str, id_2: str, origin_container: OriginContainer,
                                        thread_handler_automatic_parameter=None) -> Tuple[Dict[str, int], 
                                                                                          Dict[str, int]]:
        """
        A callback method that is meant to be called by the ThreadHandler class only. This method exists so that a
        wrapping method from the ThreadHandler class can assign threads to this method.
        :param id_1: The first identifier for which a rank dictionary should be returned
        :param id_2: The second identifier for which a rank dictionary should be returned
        :param origin_container: The origins of which tags should be considered
        :param thread_handler_automatic_parameter: This parameter will be set automatically by the ThreadHandler class
        :return: Tuple with 2 dictionaries corresponding to id_1 and id_2
        """
        # noinspection DuplicatedCode
        params: Tuple[Tuple[str, OriginContainer], Tuple[str, OriginContainer]] = (id_1, origin_container), \
                                                                                  (id_2, origin_container)
        cl: ConfigLoader = ConfigLoader.instance
        if cl.data_generation_mode_is_test and thread_handler_automatic_parameter > 0:
            threads: Set[threading.Thread] = set(
                threading.Thread(target=self.get_ranks_for_id, args=p) for p in params)
            ph: ProcessHandler = ProcessHandler.instance
            ph.execute_threads(threads=threads)
        result: Tuple[Dict[str, int], ...] = tuple(self.get_ranks_for_id(*p) for p in params)
        assert len(result) == 2
        return result[0], result[1]

    def __generate_similarities_for_multiple_origins(
            self, identifier: str, origin_containers: Tuple[OriginContainer, ...]) -> None:
        """
        :param identifier: Identifier for which to generate similarities
        :param origin_containers: Tuple that contains OriginContainers for which the similarities should be calculated
        :return: None
        """
        cl: ConfigLoader = ConfigLoader.instance
        assert cl.data_generation_mode_is_test, 'Using this method is only useful in test mode'
        to_generate: List[OriginContainer] = []
        for origin_container in origin_containers:
            df_name = generate_df_name(origin_container, is_rank=False)
            if identifier in self.__get_already_generated_ids(df_name):
                continue
            else:
                to_generate.append(origin_container)
        if len(to_generate) == 0:
            return
        args: Tuple[str, Tuple[OriginContainer, ...]] = (identifier, tuple(to_generate))
        ph: ProcessHandler = ProcessHandler.instance
        th: ProcessOrThreadHandler = ph.get_thread_handler()
        th.exec_in_pool(function=self.__thread_handled_generate_similarities_for_multiple_origins,
                        maximum=len(to_generate), args=args)

    def __thread_handled_generate_similarities_for_multiple_origins(
            self, identifier: str, origin_container_tuple: Tuple[OriginContainer, ...],
            thread_handler_automatic_parameter=None):
        """
        A callback method that is meant to be called by the ThreadHandler class only. This method exists so that a
        wrapping method from the ThreadHandler class can assign threads to this method.
        :param identifier: Identifier for which to generate similarities
        :param origin_container_tuple: Tuple that contains origin containers for which the similarities should be
        calculated
        :param thread_handler_automatic_parameter: This parameter will be set automatically by the ThreadHandler class
        :return: None
        """
        params: Tuple[Tuple[str, OriginContainer], ...] = tuple((identifier, o) for o in origin_container_tuple)
        if thread_handler_automatic_parameter > 0:
            threads: Set[threading.Thread] = set(
                threading.Thread(target=self.get_similarities_for_id, args=p) for p in params)
            ph: ProcessHandler = ProcessHandler.instance
            ph.execute_threads(threads=threads)
        else:
            _ = tuple(self.get_similarities_for_id(*p) for p in params)

    def ensure_similarities_for_pair_are_generated(self, id_1: str, id_2: str,
                                                   origin_container: OriginContainer) -> None:
        """
        Note: Only necessary in test mode
        This method checks whether entries for identifiers have been already generated and if not it does so in an
        efficient manner
        :param id_1: First identifier for which to check
        :param id_2: Second identifier for which to check
        :param origin_container: OriginContainer for which similarities should be checked / calculated
        :return: None
        """
        cl: ConfigLoader = ConfigLoader.instance
        assert cl.data_generation_mode_is_test, 'This method should only be called in test mode.'

        da: DataAccess = DataAccess.instance
        da.is_valid_identifier(identifier=id_1)
        da.is_valid_identifier(identifier=id_2)
        df_name: str
        df_name = generate_df_name(origin_container, is_rank=False)
        found_id_1: bool = id_1 in self.__get_already_generated_ids(df_name)
        found_id_2: bool = id_2 in self.__get_already_generated_ids(df_name)
        if found_id_1 and found_id_2:
            return
        _ = self.get_similarities_for_pair(id_1=id_1, id_2=id_2, origin_container=origin_container)

    def get_similarities_for_pair(
            self, id_1: str, id_2: str, origin_container: OriginContainer) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        :param id_1: The first identifier for which similarities should be returned
        :param id_2: The second identifier for which similarities should be returned
        :param origin_container: The origins of which tags should be considered
        :return: Tuple with 2 dictionaries corresponding to id_1 and id_2
        """
        da: DataAccess = DataAccess.instance
        da.is_valid_identifier(identifier=id_1)
        da.is_valid_identifier(identifier=id_2)

        cl: ConfigLoader = ConfigLoader.instance
        max_threads: int = 1
        if cl.data_generation_mode_is_test:
            df_name: str
            df_name = generate_df_name(origin_container, is_rank=False)
            if id_1 not in self.__get_already_generated_ids(df_name) and \
                    id_2 not in self.__get_already_generated_ids(df_name):
                max_threads = 2
        ph: ProcessHandler = ProcessHandler.instance
        th: ProcessOrThreadHandler = ph.get_thread_handler()
        return th.exec_in_pool(function=self.__thread_handled_get_similarities_for_pair, maximum=max_threads,
                               args=(id_1, id_2, origin_container))

    def __thread_handled_get_similarities_for_pair(
            self, id_1: str, id_2: str, origin_container: OriginContainer,
            thread_handler_automatic_parameter=None) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        A callback method that is meant to be called by the ThreadHandler class only. This method exists so that a
        wrapping method from the ThreadHandler class can assign threads to this method.
        :param id_1: The first identifier for which similarities should be returned
        :param id_2: The second identifier for which similarities should be returned
        :param origin_container: The origins of which tags should be considered
        :param thread_handler_automatic_parameter: This parameter will be set automatically by the ThreadHandler class
        :return: Tuple with 2 dictionaries corresponding to id_1 and id_2
        """
        # noinspection DuplicatedCode
        params: Tuple[Tuple[str, OriginContainer], Tuple[str, OriginContainer]] = \
            (id_1, origin_container), (id_2, origin_container)
        cl: ConfigLoader = ConfigLoader.instance
        if cl.data_generation_mode_is_test and thread_handler_automatic_parameter > 0:
            threads: Set[threading.Thread] = set(
                threading.Thread(target=self.get_similarities_for_id, args=p) for p in params)
            ph: ProcessHandler = ProcessHandler.instance
            ph.execute_threads(threads=threads)
        result = tuple(self.get_similarities_for_id(*p) for p in params)
        assert len(result) == 2
        return result[0], result[1]
