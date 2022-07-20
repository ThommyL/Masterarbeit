"""
FilterCache.py
"""

import functools
import os
from typing import Dict, List, Tuple, Optional, Set

import numpy as np
import pandas as pd
from tqdm import tqdm

from Project.AutoSimilarityCache.Caching.TagVectorCache import TagVectorCache
from Project.AutoSimilarityCache.Interface import generate_df_name, filter_and_optimize_data_tuples, \
    combined_origin_to_origin_tuple
from Project.AutoSimilarityCache.TagOriginUtils.OriginProcessing import combined_origin_to_origin_container, \
    split_origin_container_to_origin_types
from Project.AutoSimilarityCacheConfiguration.ConfigurationMethods import get_similarity_weight, get_vector, \
    get_similarity
from Project.AutoSimilarityCacheConfiguration.DataAccess import DataAccess
from Project.Utils.ConfigurationUtils.ConfigLoader import ConfigLoader
from Project.Utils.FilterCache import FILTER_CACHES_PATH
from Project.Utils.Misc.OriginContainer import OriginContainer
from Project.Utils.Misc.ProcessAndThreadHandlers import ProcessHandler
from Project.Utils.Misc.Singleton import Singleton


@Singleton
class FilterCache:
    """
    Class that speeds up dynamically filtering identifiers by using caches.
    """

    def __init__(self):
        cl: ConfigLoader = ConfigLoader.instance
        if cl.dynamic_filtering:
            self.__dir_path: str = FILTER_CACHES_PATH
            self.__file_path_dict: Dict[str, str] = dict()
            self.__dataframe_dict: Dict[str, Optional[pd.DataFrame]] = dict()

            for s in cl.combined_origins_to_consider:
                self.__file_path_dict[s] = os.path.join(self.__dir_path, 'filter_cache_' + s + '.pkl')
                self.__dataframe_dict[s] = None

            for s in cl.combined_origins_to_consider:
                # Initialization
                _ = self.__get_cache_dataframe(s)

            self.__min_length: int = 0
            self.__max_length: int = np.inf
            self.__min_sum_of_weights: float = 0
            self.__max_sum_of_weights: float = np.inf
            self.__must_contain_tags: List[str] = []
            self.__must_not_contain_tags: List[str] = []
            self.__tags_threshold_above: Dict[Tuple[any, str], float] = dict()
            self.__tags_threshold_below: Dict[Tuple[any, str], float] = dict()
            self.__min_number_of_tags_from_origin: Dict[str, int] = dict()
            self.__max_number_of_tags_from_origin: Dict[str, int] = dict()
            self.__min_sum_of_weights_of_tags_from_origin: Dict[str, float] = dict()
            self.__max_sum_of_weights_of_tags_from_origin: Dict[str, float] = dict()
            self.__must_be_created_by: List[str] = []
            self.__must_not_be_created_by: List[str] = []
            self.__must_be_created_after: int = -np.inf
            self.__must_be_created_before: int = np.inf
            self.__must_be_title: str = ''

    def __clear_caches(self):
        self.check_identifier.cache_clear()
        self.nr_of_identifiers_left.cache_clear()
        self.get_filtered_identifiers.cache_clear()
        self.nr_of_identifiers_left.cache_clear()

    @functools.lru_cache()
    def get_filtered_identifiers(self, origin_container: OriginContainer) -> Tuple[str, ...]:
        """
        :param origin_container: Filter out identifiers which do not contain at least one tag from any of the given
        origins. This parameter will not have an effect if the dynamic filtering is disabled in the config
        file.
        :return: Tuple of filtered identifier.
        """
        da: DataAccess = DataAccess.instance
        return tuple(
            i for i in da.get_ids() if self.check_identifier(
                identifier=i, origin_container=origin_container))

    def get_nr_of_tags(self, identifier: str, combined_origin: str) -> int:
        """
        :param identifier: Identifier which to check
        :param combined_origin: Combined origin for which to return the result.
        This parameter will not have an effect if the dynamic filtering is disabled in the config file.
        :return: Number of tags that the identifier has from the specified origins.
        """
        da: DataAccess = DataAccess.instance
        cl: ConfigLoader = ConfigLoader.instance
        assert da.is_valid_identifier(identifier=identifier), 'Could not find identifier in cleaned data'
        assert combined_origin in cl.combined_origins_to_consider, f'Trying to access {combined_origin}, which is ' \
                                                                   f'not specified in \'load-srd\' in the config file'
        df: pd.DataFrame = self.__dataframe_dict[combined_origin]
        return df[df['Identifier'] == identifier]['NumberOfTags'].iloc[0]

    @functools.lru_cache()
    def check_identifier(self, identifier: str, origin_container: OriginContainer) -> bool:
        """
        :param identifier: Identifier which to check
        :param origin_container: Filter out identifiers which do not contain at least one tag from any of the given
        origins. This parameter will not have an effect if the dynamic filtering is disabled in the config
        file.
        :return: True if identifier is valid, False otherwise
        """
        da: DataAccess = DataAccess.instance
        cl: ConfigLoader = ConfigLoader.instance
        if not cl.dynamic_filtering:
            return True
        assert isinstance(origin_container, OriginContainer), \
            'parameter \'origin_container\' must be an OriginContainer.'
        assert da.is_valid_identifier(identifier=identifier), 'Could not find identifier in cleaned data'
        origin_type_dict = split_origin_container_to_origin_types(origin_container)

        if len(origin_type_dict) > 1:
            for k in origin_type_dict.keys():
                if not self.check_identifier(identifier, origin_type_dict[k]):
                    return False
            return True

        df_name: str = generate_df_name(origin_container=origin_container, is_rank=False)
        assert df_name in cl.combined_origins_to_consider, f'Trying to access {df_name}, which is not specified in ' \
                                                           f'\'load-srd\' in the config file'

        if self.__must_be_title != '' and self.__must_be_title != da.get_title_for_identifier(identifier=identifier):
            return False

        if len(self.__must_not_be_created_by) != 0 or len(self.__must_be_created_by) != 0:
            creator = da.get_creator_for_identifier(identifier=identifier)
            if len(self.__must_be_created_by) != 0 and creator not in self.__must_be_created_by:
                return False
            if creator in self.__must_not_be_created_by:
                return False

        df: pd.DataFrame = self.__dataframe_dict[df_name]
        if df[df['Identifier'] == identifier]['NumberOfTags'].iloc[0] < self.__min_length:
            return False
        if df[df['Identifier'] == identifier]['NumberOfTags'].iloc[0] > self.__max_length:
            return False
        if df[df['Identifier'] == identifier]['SumOfWeights'].iloc[0] < self.__min_sum_of_weights:
            return False
        if df[df['Identifier'] == identifier]['SumOfWeights'].iloc[0] > self.__max_sum_of_weights:
            return False

        if not (len(self.__must_contain_tags) == 0 and len(self.__must_not_contain_tags) == 0 and len(
                self.__tags_threshold_above.keys()) == 0) and len(self.__tags_threshold_below.keys()) == 0:
            cur_tag_tuples: Tuple[Tuple[any, str]] = filter_and_optimize_data_tuples(
                identifier=identifier, origin_container=origin_container)
            for t in self.__must_contain_tags:
                if t not in [tt[0] for tt in cur_tag_tuples]:
                    return False
            for t in self.__must_not_contain_tags:
                if t in [tt[0] for tt in cur_tag_tuples]:
                    return False

            if len(self.__tags_threshold_above) != 0 and len(cur_tag_tuples) == 0:
                return False

            tvc: TagVectorCache = TagVectorCache.instance

            for k in self.__tags_threshold_above.keys():
                k_vector: any = get_vector(k)
                at_least_one_similar: bool = False
                for tt in cur_tag_tuples:
                    tt_vector: any = tvc.get_vector_for_tag_tuple(tag_origin_tuple=tt)
                    if get_similarity(vector_origin_tuple_1=(k_vector, k[1]),
                                      vector_origin_tuple_2=(tt_vector, tt[1])) > self.__tags_threshold_above[k]:
                        at_least_one_similar = True
                        break
                if not at_least_one_similar:
                    return False

            for k in self.__tags_threshold_below.keys():
                k_vector: any = get_vector(k)
                for tt in cur_tag_tuples:
                    tt_vector: any = tvc.get_vector_for_tag_tuple(tag_origin_tuple=tt)
                    if get_similarity(vector_origin_tuple_1=(k_vector, k[1]),
                                      vector_origin_tuple_2=(tt_vector, tt[1])) > self.__tags_threshold_below[k]:
                        return False

        if len(self.__min_number_of_tags_from_origin.keys()) != 0:
            for k in self.__min_number_of_tags_from_origin.keys():
                if df.loc[df['Identifier'] == identifier, 'Number_of_' + k + '_tags'].iloc[0] < \
                        self.__min_number_of_tags_from_origin[k]:
                    return False

        if len(self.__max_number_of_tags_from_origin.keys()) != 0:
            for k in self.__max_number_of_tags_from_origin.keys():
                if df.loc[df['Identifier'] == identifier, 'Number_of_' + k + '_tags'].iloc[0] > \
                        self.__max_number_of_tags_from_origin[k]:
                    return False

        if len(self.__min_sum_of_weights_of_tags_from_origin.keys()) != 0:
            for k in self.__min_sum_of_weights_of_tags_from_origin.keys():
                if df.loc[df['Identifier'] == identifier, 'Sum_of_weights_of_' + k + '_tags'].iloc[0] < \
                        self.__min_sum_of_weights_of_tags_from_origin[k]:
                    return False

        if len(self.__max_sum_of_weights_of_tags_from_origin.keys()) != 0:
            for k in self.__max_sum_of_weights_of_tags_from_origin.keys():
                if df.loc[df['Identifier'] == identifier, 'Sum_of_weights_of_' + k + '_tags'].iloc[0] > \
                        self.__max_sum_of_weights_of_tags_from_origin[k]:
                    return False

        if self.__must_be_created_before != np.inf and da.get_year_estimate_from_identifier(
                identifier=identifier) >= self.__must_be_created_before:
            return False

        if self.__must_be_created_after != -np.inf and da.get_year_estimate_from_identifier(
                identifier=identifier) <= self.__must_be_created_after:
            return False

        return True

    def set_rule_must_be_title(self, title: str):
        """
        :param title: Any artworks with a different title will be figured out.
        This parameter will not have an effect if the dynamic filtering is disabled in the config file.
        :return: None
        """
        if title == '':
            self.__must_be_title = ''
            return
        da: DataAccess = DataAccess.instance
        assert isinstance(title, str), 'Title must be a string'
        assert title in da.get_all_titles(), 'Title is not in database.'
        self.__clear_caches()
        self.__must_be_title = title

    def set_rule_must_be_created_after(self, min_year: int) -> None:
        """
        :param min_year: Any artworks older than the specified year will be filtered out.
        Note: year estimates are used for this purpose - see document
        This parameter will not have an effect if the dynamic filtering is disabled in the config file.
        :return: None
        """
        if min_year != -np.inf:
            assert isinstance(min_year, int), 'Parameter \'min_year\' must be an integer.'
        self.__clear_caches()
        self.__must_be_created_after = min_year

    def set_rule_must_be_created_before(self, max_year: int) -> None:
        """
        :param max_year: Any artworks newer than the specified year will be filtered out.
        Note: year estimates are used for this purpose - see document
        This parameter will not have an effect if the dynamic filtering is disabled in the config file.
        :return: None
        """
        if max_year != np.inf:
            assert isinstance(max_year, int), 'Parameter \'max_year\' must be an integer.'
        self.__clear_caches()
        self.__must_be_created_before = max_year

    def set_rule_must_be_created_by_one_of(self, creators: Tuple[str, ...]):
        """
        :param creators: Any artworks that is not created by artists in this tuple will be filtered out.
        This parameter will not have an effect if the dynamic filtering is disabled in the config file.
        :return: None
        """
        da: DataAccess = DataAccess.instance
        all_creators: Set[str] = da.get_all_creators()
        for c in creators:
            assert c in all_creators, f'Artist: {c} is not in the Dataset'
        self.__clear_caches()
        self.__must_be_created_by = creators

    def set_rule_must_not_be_created_by_one_of(self, creators: Tuple[str, ...]):
        """
        :param creators: Any artworks that is created by artists in this tuple will be filtered out.
        This parameter will not have an effect if the dynamic filtering is disabled in the config file.
        :return: None
        """
        da: DataAccess = DataAccess.instance
        all_creators: Set[str] = da.get_all_creators()
        for c in creators:
            assert c in all_creators, f'Artist: {c} is not in the Dataset'
        self.__clear_caches()
        self.__must_not_be_created_by = creators

    def set_rule_min_nr_of_tags_of_combined_origin(self, min_nr_of_tags: int, combined_origin: str) -> None:
        """
        :param min_nr_of_tags: Minimal number of tags. If an identifier has fewer tags belonging to the specified
        origins, it will be filtered out. This parameter will not have an effect if the dynamic filtering is
        disabled in the config file.
        :param combined_origin: Combined origin for which the rule should apply
        :return: None
        """
        assert isinstance(min_nr_of_tags, int), 'Parameter \'min_nr_of_tags\' must be an integer.'
        assert isinstance(combined_origin, str), 'Parameter \'combined_origin\' must be a str.'
        assert min_nr_of_tags >= 0, 'Parameter \'min_nr_of_tags\' must be positive.'
        cl: ConfigLoader = ConfigLoader.instance
        assert combined_origin in cl.all_unique_combined_origins, 'Parameter \'combined_origin\' must be a valid ' \
                                                                  'combined origin.'
        self.__clear_caches()
        self.__min_number_of_tags_from_origin[combined_origin] = min_nr_of_tags

    def set_rule_max_nr_of_tags_of_combined_origin(self, max_nr_of_tags: int, combined_origin: str) -> None:
        """
        :param max_nr_of_tags: Maximum number of tags. If an identifier has more tags belonging to the specified
        origins, it will be filtered out. This parameter will not have an effect if the dynamic filtering is
        disabled in the config file.
        :param combined_origin: Combined origin for which the rule should apply
        :return: None
        """
        assert isinstance(max_nr_of_tags, int), 'Parameter \'max_nr_of_tags\' must be an integer.'
        assert isinstance(combined_origin, str), 'Parameter \'combined_origin\' must be a str.'
        assert max_nr_of_tags >= 0, 'Parameter \'max_nr_of_tags\' must be positive.'
        cl: ConfigLoader = ConfigLoader.instance
        assert combined_origin in cl.currently_possible_combined_origins, \
            'Parameter \'combined_origin\' must be a valid combined origin.'
        self.__clear_caches()
        self.__max_number_of_tags_from_origin[combined_origin] = max_nr_of_tags

    def set_rule_min_sum_of_weights_of_tags_of_combined_origin(
            self, min_sum_of_weights: int, combined_origin: str) -> None:
        """
        :param min_sum_of_weights: Minimal sum of weights of tags belonging to the specified origins. If an identifier
        has a smaller sum of weights than specified it will be filtered out. This parameter will not have an
        effect if the dynamic filtering is disabled in the config file.
        :param combined_origin: Combined origin for which the rule should apply
        :return: None
        """
        if isinstance(min_sum_of_weights, int):
            min_sum_of_weights: float = float(min_sum_of_weights)
        assert isinstance(min_sum_of_weights, float), 'Parameter \'min_sum_of_weights\' must be a float.'
        assert isinstance(combined_origin, str), 'Parameter \'combined_origin\' must be a str.'
        assert min_sum_of_weights >= 0, 'Parameter \'min_sum_of_weights\' must be positive.'
        cl: ConfigLoader = ConfigLoader.instance
        assert combined_origin in cl.all_unique_combined_origins, 'Parameter \'combined_origin\' must be a valid ' \
                                                                  'combined origin.'
        self.__clear_caches()
        self.__min_sum_of_weights_of_tags_from_origin[combined_origin] = min_sum_of_weights

    def set_rule_max_sum_of_weights_of_tags_of_combined_origin(
            self, max_sum_of_weights: int, combined_origin: str) -> None:
        """
        :param max_sum_of_weights: Maximum sum of weights of tags belonging to the specified origins. If an identifier
        has a larger sum of weights than specified it will be filtered out. This parameter will not have an
        effect if the dynamic filtering is disabled in the config file.
        :param combined_origin: Combined origin for which the rule should apply
        :return: None
        """
        if isinstance(max_sum_of_weights, int):
            max_sum_of_weights: float = float(max_sum_of_weights)
        assert isinstance(max_sum_of_weights, float), 'Parameter \'max_sum_of_weights\' must be a float.'
        assert isinstance(combined_origin, str), 'Parameter \'combined_origin\' must be a str.'
        assert max_sum_of_weights >= 0, 'Parameter \'max_sum_of_weights\' must be positive.'
        cl: ConfigLoader = ConfigLoader.instance
        assert combined_origin in cl.all_unique_combined_origins, 'Parameter \'combined_origin\' must be a valid ' \
                                                                  'combined origin.'
        self.__clear_caches()
        self.__max_sum_of_weights_of_tags_from_origin[combined_origin] = max_sum_of_weights

    def set_rule_min_length(self, min_length: int) -> None:
        """
        :param min_length: Minimal number of tags. If an identifier has less tags belonging to the specified origins, it
        will be filtered out. This parameter will not have an effect if the dynamic filtering is disabled in
        the config file.
        :return: None
        """
        assert isinstance(min_length, int), 'Parameter \'min_length\' must be an integer.'
        assert min_length >= 0, 'Parameter \'min_length\' must be positive.'
        self.__clear_caches()
        self.__min_length = min_length

    def set_rule_max_length(self, max_length: int) -> None:
        """
        :param max_length: Maximal number of tags. If an identifier has more tags belonging to the specified origins, it
        will be filtered out. This parameter will not have an effect if the dynamic filtering is disabled in
        the config file.
        :return: None
        """
        if max_length != np.inf:
            assert isinstance(max_length, int), 'Parameter \'min_length\' must be an integer.'
        assert max_length >= 0, 'Parameter \'min_length\' must be positive.'
        self.__clear_caches()
        self.__max_length = max_length

    def set_rule_min_sum_of_weights(self, min_sum_of_weights: float) -> None:
        """
        :param min_sum_of_weights: Minimal value of the combined weights of tags. If the sum of weights of tags from
        requested origins is lower than this value it will be filtered out. This parameter will not have an
        effect if the dynamic filtering is disabled in the config file.
        :return: None
        """
        if isinstance(min_sum_of_weights, int):
            min_sum_of_weights: float = float(min_sum_of_weights)
        assert isinstance(min_sum_of_weights, float), 'Parameter \'min_sum_of_weights\' must be a float.'
        assert min_sum_of_weights >= 0, 'Parameter \'min_sum_of_weights\' must be positive.'
        self.__clear_caches()
        self.__min_sum_of_weights = min_sum_of_weights

    def set_rule_max_sum_of_weights(self, max_sum_of_weights: float) -> None:
        """
        :param max_sum_of_weights: maximal value of the sum of weights of tags. If the sum of weights of tags from
        requested origins is higher than this value it will be filtered out. This parameter will not have an
        effect if the dynamic filtering is disabled in the config file.
        :return: None
        """
        if isinstance(max_sum_of_weights, int):
            max_sum_of_weights: float = float(max_sum_of_weights)
        assert isinstance(max_sum_of_weights, float), 'Parameter \'max_sum_of_weights\' must be a float.'
        assert max_sum_of_weights >= 0, 'Parameter \'max_sum_of_weights\' must be positive.'
        self.__clear_caches()
        self.__max_sum_of_weights = max_sum_of_weights

    def set_rule_must_contain_tags(self, tags: Tuple[str, ...]) -> None:
        """
        :param tags: A tuple of tags, for which identifiers will be filtered out if its tags do not contain at least one
        of the specified tags. This parameter will not have an effect if the dynamic filtering is disabled in
        the config file.
        :return: None
        """
        assert isinstance(tags, tuple), 'Parameter \'tags\' must be a tuple of strings.'
        cl: ConfigLoader = ConfigLoader.instance
        assert cl.dynamic_filtering, \
            'The parameter \'dynamic-filtering\' in the config file is set to false.'
        for t in tags:
            assert isinstance(t, str), 'Parameter \'tags\' must be a tuple of strings.'
        self.__clear_caches()
        self.__must_contain_tags = tags

    def set_rule_must_not_contain_tags(self, tags: Tuple[str, ...]) -> None:
        """
        :param tags: A tuple of tags, for which identifiers will be filtered out if its tags contain one of the
        specified tags. This parameter will not have an effect if the dynamic filtering is disabled in
        the config file.
        :return: None
        """
        assert isinstance(tags, tuple), 'Parameter \'tags\' must be a tuple of strings.'
        cl: ConfigLoader = ConfigLoader.instance
        assert cl.dynamic_filtering, \
            'The parameter \'dynamic-filtering\' in the config file is set to false.'
        for t in tags:
            assert isinstance(t, str), 'Parameter \'tags\' must be a tuple of strings.'
        self.__clear_caches()
        self.__must_not_contain_tags = tags

    # noinspection DuplicatedCode
    def set_rule_must_be_similar_to_tags(self, tags_threshold: Dict[Tuple[any, str], float]) -> None:
        """
        :param tags_threshold: A dictionary containing Tuples[tag, origin] as keys and a value between -1 and 1 as
        minimum required similarity. Artworks that do not have at least one tag which fulfills the requirement will be
        filtered out. Note that the second entry of the key tuple is a origin: This specifies what origin the value
        should be treated like. This parameter will not have an effect if the dynamic filtering is disabled in the
        config file.
        :return: None
        """
        assert isinstance(tags_threshold, dict), 'Parameter \'tags_threshold\' must be a dictionary.'
        cl: ConfigLoader = ConfigLoader.instance
        assert cl.dynamic_filtering, \
            'The parameter \'dynamic-filtering\' in the config file is set to false.'
        for k in tags_threshold.keys():
            assert isinstance(k, tuple), 'Keys must be tuples'
            assert isinstance(k[1], str), 'Keys must reference an origin at index 1, which must be a string'

        for v in tags_threshold.values():
            assert isinstance(v, float), 'Values must be floats'
            assert -1 <= v <= 1, 'Values must be greater or equal -1 and smaller or equal 1.'
        self.__clear_caches()
        self.__tags_threshold_above = tags_threshold

    # noinspection DuplicatedCode
    def set_rule_must_not_be_similar_to_tags(self, tags_threshold: Dict[Tuple[any, str], float]) -> None:
        """
        :param tags_threshold: A dictionary containing Tuples[tag, origin] as keys and a value between -1 and 1 as
        maximum allowed similarity. Artworks that do not have at least one tag which fulfills the requrement will be
        filtered out.  Note that the second entry of the key tuple is a origin: This specifies what origin the value
        should be treated like. This parameter will not have an effect if the dynamic filtering is disabled in the
        config file.
        :return: None
        """
        assert isinstance(tags_threshold, dict), 'Parameter \'tags_threshold\' must be a dictionary.'
        cl: ConfigLoader = ConfigLoader.instance
        assert cl.dynamic_filtering, \
            'The parameter \'dynamic-filtering\' in the config file is set to false.'
        for k in tags_threshold.keys():
            assert isinstance(k, tuple), 'Keys must be tuples'
            assert isinstance(k[1], str), 'Keys must reference an origin at index 1, which must be a string'

        for v in tags_threshold.values():
            assert isinstance(v, float), 'Values must be floats'
            assert -1 <= v <= 1, 'Values must be greater or equal -1 and smaller or equal 1.'
        self.__clear_caches()
        self.__tags_threshold_below = tags_threshold

    def __get_cache_dataframe(self, combined_origin: str) -> pd.DataFrame:
        """
        :param combined_origin: Origin(s) for which to return the dataframe
        :return: Dataframe according to specified origins
        """
        if self.__dataframe_dict[combined_origin] is not None:
            return self.__dataframe_dict[combined_origin]

        if not os.path.exists(self.__file_path_dict[combined_origin]):
            ph: ProcessHandler = ProcessHandler.instance
            while True:
                if not ph.acquire_data_generation_lock():
                    # If this was generated in the meanwhile
                    if os.path.exists(self.__file_path_dict[combined_origin]):
                        break
                    df: pd.DataFrame = self.__get_empty_dataframe()
                    da: DataAccess = DataAccess.instance
                    for ind, row in tqdm(df.iterrows(), total=df.shape[0],
                                         desc=f'Generating tuple length cache for origin: {combined_origin}'):
                        cur_identifier = row['Identifier']
                        origin_container: OriginContainer = OriginContainer(
                            combined_origin_to_origin_tuple(combined_origin))
                        df.loc[ind, 'NumberOfTags'] = len(
                            filter_and_optimize_data_tuples(
                                identifier=cur_identifier, origin_container=origin_container))
                        df.loc[ind, 'SumOfWeights'] = sum(
                            tuple(
                                get_similarity_weight(identifier=cur_identifier, tag_origin_tuple=tt)
                                for tt in filter_and_optimize_data_tuples(
                                    identifier=cur_identifier, origin_container=origin_container)))

                        cl: ConfigLoader = ConfigLoader.instance

                        for co in cl.all_unique_combined_origins:
                            df.loc[ind, 'Number_of_' + co + '_tags'] = len(
                                tuple(filter_and_optimize_data_tuples(
                                    cur_identifier, combined_origin_to_origin_container(co))))
                            df.loc[ind, 'Sum_of_weights_of_' + co + '_tags'] = sum(
                                da.get_weight_for_identifier_tag_tuple(identifier=cur_identifier, tag_origin_tuple=t)
                                for t in tuple(
                                    filter_and_optimize_data_tuples(
                                        cur_identifier, combined_origin_to_origin_container(co))))
                    df.to_pickle(self.__file_path_dict[combined_origin])
            ph.release_data_generation_lock()
        self.__dataframe_dict[combined_origin] = pd.read_pickle(self.__file_path_dict[combined_origin])
        return self.__dataframe_dict[combined_origin]

    def __get_empty_dataframe(self) -> pd.DataFrame:
        """
        :return: An empty template dataframe
        """
        empty_path: str = os.path.join(self.__dir_path, 'empty_filter_cache_dataframe.pkl')
        if not os.path.exists(empty_path):
            da: DataAccess = DataAccess.instance
            all_rows: List[Dict] = []
            cl: ConfigLoader = ConfigLoader.instance

            number_of_tags_per_origin_columns: Tuple[str, ...] = tuple(
                'Number_of_' + o + '_tags' for o in cl.possible_origins)
            sum_of_weight_of_tags_per_origin_columns: Tuple[str, ...] = tuple(
                'Sum_of_weights_of_' + o + '_tags' for o in cl.possible_origins)

            for i in tqdm(da.get_ids(), desc="Generating template dataframe for empty cache"):
                cur: Dict[str, any] = dict()
                cur['Identifier'] = i
                cur['NumberOfTags'] = 0
                for c in number_of_tags_per_origin_columns:
                    cur[c] = 0
                for c in sum_of_weight_of_tags_per_origin_columns:
                    cur[c] = 0.0
                all_rows.append(cur)

            columns = tuple(['Identifier', 'NumberOfTags'] + list(number_of_tags_per_origin_columns) +
                            list(sum_of_weight_of_tags_per_origin_columns))
            df: pd.DataFrame = pd.DataFrame(all_rows, columns=columns)
            df = df.reset_index()

            type_dictionary: Dict[str, type] = dict()
            type_dictionary['Identifier'] = str
            type_dictionary['NumberOfTags'] = int
            for c in number_of_tags_per_origin_columns:
                type_dictionary[c] = int
            for c in sum_of_weight_of_tags_per_origin_columns:
                type_dictionary[c] = float
            df = df.astype(type_dictionary)
            df.to_pickle(empty_path)
        return pd.read_pickle(empty_path)

    def is_active(self) -> bool:
        """
        :return: True if any identifiers are filtered, False otherwise.
        """
        cl: ConfigLoader = ConfigLoader.instance

        return cl.dynamic_filtering and (
                self.__min_length != 0 or self.__max_length != np.inf or len(self.__must_contain_tags) > 0 or
                len(self.__must_not_contain_tags) > 0 or len(self.__tags_threshold_above.keys()) > 0 or
                len(self.__tags_threshold_below.keys()) > 0 or
                len(self.__min_number_of_tags_from_origin.keys()) > 0 or
                len(self.__max_number_of_tags_from_origin.keys()) > 0 or
                len(self.__min_sum_of_weights_of_tags_from_origin.keys()) > 0 or
                len(self.__max_sum_of_weights_of_tags_from_origin.keys()) > 0 or
                len(self.__must_be_created_by) > 0 or
                len(self.__must_not_be_created_by) > 0 or
                self.__min_sum_of_weights != 0 or
                self.__max_sum_of_weights != np.inf or
                self.__must_be_created_before != np.inf or self.__must_be_created_after != -np.inf or
                self.__must_be_title != ''
        )

    def __str__(self) -> str:
        if not self.is_active():
            return 'No filters active.'
        result: str = ''
        if self.__min_length != 0:
            result += f'The minimum amount of tags from requested origins is: {self.__min_length}\n'
        if self.__max_length != np.inf:
            result += f'The maximum amount of tags from requested origins is: {self.__max_length}\n'
        if self.__min_sum_of_weights != 0:
            result += f'The minimum sum of weights of tags from requested origins is: {self.__min_sum_of_weights}\n'
        if self.__max_sum_of_weights != np.inf:
            result += f'The maximum sum of weights of tags from requested origins is: {self.__max_sum_of_weights}\n'
        for t in self.__must_contain_tags:
            result += f'The result must contain the tag: {t}\n'
        for t in self.__must_not_contain_tags:
            result += f'The result must not contain the tag: {t}\n'
        for k in self.__tags_threshold_above.keys():
            result += f'The result must contain at least one tag that has a similarity >= ' \
                      f'{self.__tags_threshold_above[k]} to "{k}"\n'
        for k in self.__tags_threshold_above.keys():
            result += f'The result must not contain any tags that have a similarity >= ' \
                      f'{self.__tags_threshold_below[k]} to "{k}"\n'
        for k in self.__min_number_of_tags_from_origin.keys():
            result += f'The minimum amount of tags from the origin "{k}" is ' \
                      f'{self.__min_number_of_tags_from_origin[k]}\n'
        for k in self.__max_number_of_tags_from_origin.keys():
            result += f'The maximum amount of tags from the origin "{k}" is ' \
                      f'{self.__max_number_of_tags_from_origin[k]}\n'
        for k in self.__min_sum_of_weights_of_tags_from_origin:
            result += f'The minimum sum of weights from the origin "{k}" is ' \
                      f'{self.__min_sum_of_weights_of_tags_from_origin[k]}\n'
        for k in self.__max_sum_of_weights_of_tags_from_origin:
            result += f'The maximum sum of weights from the origin "{k}" is ' \
                      f'{self.__max_sum_of_weights_of_tags_from_origin[k]}\n'
        if len(self.__must_be_created_by) > 0:
            result += f'The artwork must be created by one of the following artists:\n    '
        for a in self.__must_be_created_by:
            result += f'{a}, '
        if len(self.__must_be_created_by) > 0:
            result = result[:-2]
            result += '\n'
        if len(self.__must_not_be_created_by) > 0:
            result += f'The artwork must not be created by one of the following artists:\n    '
        for a in self.__must_be_created_by:
            result += f'{a}, '
        if len(self.__must_not_be_created_by) > 0:
            result = result[:-2]
            result += '\n'
        if self.__must_be_created_after != -np.inf:
            result += f'The artwork must be created after {self.__must_be_created_after}\n'
        if self.__must_be_created_before != np.inf:
            result += f'The artwork must be created before {self.__must_be_created_before}\n'
        if self.__must_be_title != '':
            result += f'The artwork must have the title {self.__must_be_title}\n'
        return result

    @functools.lru_cache()
    def nr_of_identifiers_left(self, origin_container: OriginContainer) -> int:
        """
        :param origin_container: The origins which to consider.
        :return: The number of ids that are still left, after applying the filtering
        """
        da: DataAccess = DataAccess.instance
        if not self.is_active():
            return len(da.get_ids())
        return len([i for i in da.get_ids() if self.check_identifier(
            identifier=i, origin_container=origin_container)])

    def reset_filters(self) -> None:
        """
        Resets the filters
        :return: None
        """
        self.__min_length = 0
        self.__max_length = np.inf
        self.__must_contain_tags = dict()
        self.__must_not_contain_tags = dict()
        self.__tags_threshold_above = dict()
        self.__tags_threshold_below = dict()
        self.__min_number_of_tags_from_origin = dict()
        self.__max_number_of_tags_from_origin = dict()
        self.__min_sum_of_weights_of_tags_from_origin = dict()
        self.__max_sum_of_weights_of_tags_from_origin = dict()
        self.__must_be_created_by = []
        self.__must_not_be_created_by = []
        self.__must_be_created_after = -np.inf
        self.__must_be_created_before = np.inf
        self.__must_be_title = ''
        self.__clear_caches()
