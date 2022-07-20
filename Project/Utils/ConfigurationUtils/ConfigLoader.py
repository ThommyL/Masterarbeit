"""
ConfigLoader.py
"""

import json
import os
from typing import Dict, Tuple, Set, Optional

import psutil
import torch

from Project.AutoSimilarityCacheConfiguration.ConfigurationMethods import validate_configuration, \
    get_possible_combined_origins, get_possible_origins, get_generally_possible_combined_origins, \
    get_type_equally_groups
from Project.Utils.Misc.OriginContainer import OriginContainer
from Project.Utils.Misc.Singleton import Singleton


@Singleton
class ConfigLoader:
    """
    Loads and checks the config file, but also provides system information
    """

    def __init__(self):
        self.__active_origins: Optional[Tuple[str, ...]] = None

        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'config.json')
        try:
            with open(config_path) as f:
                self.__config: Dict[str, any] = json.load(f)
        except FileNotFoundError as e:
            print(f'Execute the method \'init_config_file()\' in utils_misc.py first.')
            raise e

        # General checks:
        for key in [
            'first-start', 'gpu-device', 'data-generation-mode', 'single-thread-mode', 'combined-origins-to-consider',
            'threads-per-core', 'n-cores', 'dynamic-filtering', 'overbook-threads', 'enable-debug-output',
            'enable-process-and-thread-handler-debug-output', 'enable-similarity-rank-database-debug-output',
            'enable-matching-debug-output', 'average-similarity-weighted-random-sample-size', 'matching-mode'
        ]:
            assert key in self.__config, f'Missing parameter {key} in config.'

        # Specific checks:
        assert self.__config['first-start'] in [True, False], 'Parameter \'first-start\' must be set to ' \
                                                              'either true or false'
        assert isinstance(self.__config['gpu-device'], int), 'Parameter \'gpu-device\' requires value of type int'
        assert self.__config['gpu-device'] <= torch.cuda.device_count(), \
            f'Parameter \'gpu-device\' was {self.__config["gpu-device"]}, ' \
            f'but only {torch.cuda.device_count()} are available.'
        assert self.__config['gpu-device'] >= -1, f'Parameter \'gpu-device\' must be >= -1'
        assert self.__config['data-generation-mode'] in ['prod', 'test'], \
            'Parameter \'data-generation-mode\' must be set to either \'prod\' or \'test\''
        assert self.__config['single-thread-mode'] in [True, False], 'Parameter \'single-thread-mode\'' \
                                                                     ' must be set to either true or false'
        assert isinstance(self.__config['threads-per-core'], int)
        assert 1 <= self.__config['threads-per-core'] <= 10, 'A minimum of 1 and a maximum of 10 threads per core ' \
                                                             'is allowed.'
        assert self.__config['dynamic-filtering'] in [True, False], 'Parameter \'first-start\' must be set to ' \
                                                                    'either true or false'
        assert self.__config['overbook-threads'] in [True, False], 'Parameter \'overbook-threads\' must be set to ' \
                                                                   'either true or false'
        assert self.__config['enable-debug-output'] in [True, False], 'Parameter \'enable-debug-output\' must be set ' \
                                                                      'to either true or false'
        assert self.__config['enable-process-and-thread-handler-debug-output'] in [True, False], \
            'Parameter \'enable-process-and-thread-handler-debug-output\' must be set to either true or false'
        assert self.__config['enable-similarity-rank-database-debug-output'] in [True, False], \
            'Parameter \'enable-similarity-rank-database-debug-output\' must be set to either true or false'
        assert self.__config['enable-matching-debug-output'] in [True, False], \
            'Parameter \'enable-matching-debug-output\' must be set to either true or false'
        assert isinstance(self.__config['n-cores'], int), 'Parameter \'n-cores\' requires value of type int'
        assert self.__config['n-cores'] > 0, 'Parameter \'n-cores\' must be a positive int'
        # noinspection PyArgumentEqualDefault
        assert self.__config['n-cores'] < psutil.cpu_count(logical=True), \
            f'Trying to use more cores than are available ({psutil.cpu_count(logical=True)})'

        assert isinstance(self.__config['average-similarity-weighted-random-sample-size'], dict)
        sample_size_dict: Dict = self.__config['average-similarity-weighted-random-sample-size']
        for k in sample_size_dict.keys():
            assert isinstance(k, str), \
                'Parameter \'average-similarity-weighted-random-sample-size\' requires keys of type int'
            assert k in get_generally_possible_combined_origins(), \
                'Parameter \'average-similarity-weighted-random-sample-size\' requires keys that is a combined ' \
                'origin according to the method \'get_generally_possible_combined_origins\' in ConfigurationMethods.py'
            assert isinstance(sample_size_dict[k], int), \
                'Parameter \'average-similarity-weighted-random-sample-size\' requires values of type int'
            assert sample_size_dict[k] > 0, \
                'Parameter \'average-similarity-weighted-random-sample-size\' must be larger than 0'
        assert isinstance(self.__config['matching-mode'], str), 'Parameter \'matching-mode\' must be of type str'
        assert self.__config['matching-mode'] in ['smooth', 'experimental'], 'Parameter \'matching-mode\' must be ' \
                                                                             'either \'smooth\' or \'experimental\''
        self.__database_generation_in_progress: bool = False
        self.__data_cleaning_in_progress: bool = False

        possible: Set[str] = set()
        for combined_origin in self.combined_origins_to_consider:
            for origin in self.possible_origins:
                if origin in combined_origin.split('&'):
                    possible.add(origin)

        # Getting the right order again:
        self.__active_origins: Tuple[str, ...] = tuple(o for o in get_possible_origins() if o in possible)

        for elem in self.__config['combined-origins-to-consider']:
            assert elem in self.currently_possible_combined_origins, \
                f'Invalid value in combined-origins-to-consider: {elem}'

        validate_configuration(self.__active_origins,
                               active_combined_origins=self.combined_origins_to_consider)

        self.__origin_type_weight_dict: Optional[Dict[str, float]] = None

    @property
    def origin_type_group_weight_dict(self):
        """
        :return: The dictionary that defines how to weight the different type groups (see ConfigurationMethods.py)
        """
        return self.__origin_type_weight_dict

    @origin_type_group_weight_dict.setter
    def origin_type_group_weight_dict(self, value) -> None:
        """
        :param value: A dictionary that defines how to weight the different type groups (see ConfigurationMethods.py)
        :return: None
        """
        weight_sum = 0
        for k in value.keys():
            assert k in get_type_equally_groups()
            assert 0 <= value[k] <= 1
            weight_sum += value[k]
        assert weight_sum == 1
        self.__origin_type_weight_dict = value

    @property
    def n_cores(self) -> int:
        """
        :return: True if the corresponding parameter in the config file is True, False otherwise
        """
        return self.__config['n-cores']

    @property
    def debug_output_enabled(self):
        """
        :return: True if the corresponding parameter in the config file is True, False otherwise
        """
        return self.__config['enable-debug-output']

    @property
    def process_and_thread_handler_debug_output_enabled(self):
        """
        :return: True if the corresponding parameter in the config file is True, False otherwise
        """
        return self.debug_output_enabled and self.__config['enable-process-and-thread-handler-debug-output']

    @property
    def similarity_rank_database_debug_output_enabled(self):
        """
        :return: True if the corresponding parameter in the config file is True, False otherwise
        """
        return self.debug_output_enabled and self.__config['enable-similarity-rank-database-debug-output']

    @property
    def matching_debug_output_enabled(self):
        """
        :return: True if the corresponding parameter in the config file is True, False otherwise
        """
        return self.debug_output_enabled and self.__config['enable-matching-debug-output']

    @property
    def dynamic_filtering(self) -> bool:
        """
        :return: True if the corresponding parameter in the config file is True, False otherwise
        """
        return self.__config['dynamic-filtering']

    @property
    def number_of_threads_to_use(self) -> int:
        """
        :return: Value of the corresponding parameter in the config file
        """
        return self.__config['threads-per-core']

    @property
    def database_generation_in_progress(self) -> bool:
        """
        :return: True if there is no certainty that no data will be generated
        """
        return self.__database_generation_in_progress or self.data_generation_mode_is_test

    @database_generation_in_progress.setter
    def database_generation_in_progress(self, val: bool) -> None:
        """
        :param val: True or False
        :return: None
        """
        assert val in [True, False], 'Only boolean values are accepted'
        self.__database_generation_in_progress = val

    @property
    def gpu_device(self) -> int:
        """
        :return: The Gpu device specified in the config file
        """
        return self.__config['gpu-device']

    @property
    def data_generation_mode_is_test(self) -> bool:
        """
        :return: True if the corresponding parameter in the config file is "Test", False otherwise
        """
        return self.__config['data-generation-mode'] == 'test'

    @property
    def single_thread_mode(self) -> bool:
        """
        :return: True if the corresponding parameter in the config file is True, False otherwise
        """
        return self.__config['single-thread-mode']

    @property
    def first_start(self) -> bool:
        """
        :return: True if the corresponding parameter in the config file is True, False otherwise
        """
        return self.__config['first-start']

    @property
    def combined_origins_to_consider(self) -> Tuple[str, ...]:
        """
        :return: The combined origins given in the config file
        """
        return tuple(self.__config['combined-origins-to-consider'])

    @property
    def parameter_names(self) -> Tuple[str, ...]:
        """
        :return: The parameter names of the config file
        """
        return tuple(self.__config.keys())

    @property
    def currently_possible_combined_origins(self) -> Tuple[str, ...]:
        """
        :return: The dictionary names that are currently active.
        """
        return get_possible_combined_origins(self.active_origins)

    @property
    def all_unique_combined_origins(self) -> Tuple[str, ...]:
        """
        :return: The dictionary names that can appear considering all valid combined origins.
        """
        return get_possible_combined_origins(None)

    @property
    def possible_dictionary_names(self) -> Tuple[str, ...]:
        """
        :return: The dictionary names that are possible in general (not just the active ones)
        """
        return get_generally_possible_combined_origins()

    @property
    def possible_origins(self) -> Tuple[str, ...]:
        """
        :return: The origins that are possible in general (not just the active ones)
        """
        return get_possible_origins()

    @property
    def active_origins(self) -> Tuple[str, ...]:
        """
        :return: Origins that are active in the right order to combine them to create a combined origin
        """
        return self.__active_origins

    @property
    def overbook_threads(self) -> bool:
        """
        :return: True if the corresponding parameter in the config file is True, False otherwise
        """
        return self.__config['overbook-threads'] and not self.single_thread_mode

    @property
    def data_cleaning_in_progress(self) -> bool:
        """
        :return: Value of data_cleaning_in_progress
        """
        return self.__data_cleaning_in_progress

    @property
    def matching_experimental_mode(self) -> bool:
        """
        :return: True if matching mode is set to "experimental", False otherwise
        """
        return self.__config['matching-mode'] == 'experimental'

    @data_cleaning_in_progress.setter
    def data_cleaning_in_progress(self, val: bool) -> None:
        """
        :param val: True or False
        :return: None
        """
        assert val in [True, False], 'Only boolean values are accepted'
        self.__data_cleaning_in_progress = val

    def weighted_average_similarity_random_sample_size(self, combined_origin: str) -> int:
        """
        :param combined_origin for which the value should be returned
        :return: The amount of random samples that should be used to calculate the average similarity of a tag, given
        the specified origin
        """
        assert combined_origin in self.combined_origins_to_consider, \
            'Value must be a combined origin that is currently loaded'
        return self.__config['average-similarity-weighted-random-sample-size'][combined_origin]

    def get_origin_container_with_active_origins(self) -> OriginContainer:
        """
        :return: An OriginContainer where all active origins are set to True
        """
        return OriginContainer(origins=self.active_origins)

    def get_origin_container_with_possible_origins(self) -> OriginContainer:
        """
        :return: An OriginContainer where all possible origins are set to True
        """
        return OriginContainer(origins=self.possible_origins)
