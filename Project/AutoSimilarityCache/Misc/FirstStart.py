"""
FirstStart.py
"""

import os
import time
import warnings
from typing import Optional

from Project.AutoSimilarityCache.Caching.TagVectorCache import TagVectorCache
from Project.AutoSimilarityCache.Misc.Misc import check_constraints
from Project.Utils.IconclassCache.IconclassCache import IconclassCacheGenerationFailedException, \
    NumberOfTriesExceededException


def first_start() -> None:
    """
    A method that should be executed on first start. First the config file is generated with standard parameters. Then
    the similarity databases are generated. Finally the config file is rewritten so that the first start parameter is
    then set to false.
    :return: None
    """

    assert os.path.exists(
        os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Dataset_Tasks', 'cleaned_dataframe.pkl')), \
        '"cleaned_dataframe.pkl" does not exist in the folder "Dataset_Tasks". Do you need to run "DatasetLoader.py" ' \
        'and "DatasetCleaning.py"?'

    __init_config_file()

    from Project.AutoSimilarityCache.Caching.SimilarityRankDatabase import SimilarityRankDatabase
    from Project.AutoSimilarityCache.Caching.TagAverageSimilarityCache import TagAverageSimilarityCache
    from Project.Utils.IconclassCache.IconclassCache import IconclassCache
    from Project.Utils.ConfigurationUtils.ConfigLoader import ConfigLoader
    from Project.Utils.FilterCache.FilterCache import FilterCache
    from Project.AutoSimilarityCacheConfiguration.DataAccess import DataAccess

    cl: ConfigLoader = ConfigLoader.instance
    assert not cl.data_generation_mode_is_test, 'Parameter \'data-generation-mode\' in config file must be set to ' \
                                                '\'prod\'.'
    cl.database_generation_in_progress = True
    assert cl.first_start, 'First start is set to false. This crash should prevent unintentional overwriting of ' \
                           'caches. If you want to regenerate the caches, please set the parameter \'first-start\' in' \
                           ' the config file to true.'

    check_constraints()

    warnings.warn(
        'Starting generation of database, this can take a very long time. You may still stop the program within '
        '3 minutes.')
    time.sleep(180)
    warnings.warn('Started')

    _ = DataAccess.instance

    _ = TagVectorCache.instance

    _ = TagAverageSimilarityCache.instance

    srd: SimilarityRankDatabase = SimilarityRankDatabase.instance
    srd.generate_similarity_and_rank_caches()

    _ = FilterCache.instance

    exception_raised: Optional[Exception] = None
    try:
        _ = IconclassCache.instance
    except (IconclassCacheGenerationFailedException, NumberOfTriesExceededException) as exception:
        exception_raised = exception

    __rewrite_config_file_param(param='first-start', set_to='false', is_string=False)
    cl.database_generation_in_progress = False
    print('Cache Generation was successful.')

    if exception_raised is not None:
        print("An Error occured when generating files for the class IconclassCache."
              "This process depends on external resources and will fail if these are not provided anymore or have "
              "changed in a significant way. "
              "This only affects the IconclassComparisonVisulaization, which is used as a method of validation only. "
              "Therefore the rest of the program is not affected.")
        print('The following exception was encountered while generating the IconclassCache:')
        raise exception_raised


def __init_config_file() -> None:
    """
    Writes a config file with standard parameters.
    :return: None
    """
    config_path: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.json')
    if os.path.exists(config_path):
        return
    config_template_path: str = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                                             'Utils/ConfigurationUtils/config_template.json')
    with open(config_template_path, 'r+') as template:
        with open(config_path, 'w+') as target:
            for line in template:
                print(line, file=target, end='')


def __rewrite_config_file_param(param: str, set_to: str, is_string: bool) -> None:
    """
    Rewrites the config file, to set specific parameters. This is only allowed when first start is true.
    :param param: Parameter of the config file that is to be changed
    :param set_to: Value that should be written into the config file for the specified parameter
    :param is_string: True if quotation marks should be put around the parameter, False otherwise
    :return: None
    """
    assert isinstance(param, str), 'Parameter \'param\' must be str'
    assert isinstance(set_to, str), 'Parameter \'set_to\' must be str'

    from Project.Utils.ConfigurationUtils.ConfigLoader import ConfigLoader

    cl: ConfigLoader = ConfigLoader.instance

    assert cl.first_start, \
        'Automatically rewriting the config file is only possible on the first start.'
    config_path: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.json')
    assert os.path.exists(config_path), f'There is no config file at {config_path}.' \
                                        f'You may want to execute method \'first_start\'.'
    config_temporary_path = os.path.join(os.path.dirname(__file__), 'config_temporary.json')
    assert param in cl.parameter_names, 'Invalid parameter name'
    number_of_lines: int = 0
    with open(config_path, 'r+') as config:
        for _ in config:
            number_of_lines += 1
    line_edited: bool = False
    cur_line: int = 0
    with open(config_path, 'r+') as config:
        with open(config_temporary_path, 'w+') as temporary:
            for line in config:
                cur_line += 1
                if param in line:
                    assert not line_edited, 'Specified line appeared twice.'
                    line_edited = True
                    end_with: str = ',\n'
                    if cur_line == number_of_lines - 1:
                        end_with = '\n'
                    if is_string:
                        set_to = f"{set_to}"
                    print(f'  "{param}": {set_to}' + end_with, file=temporary, end='')
                else:
                    print(line, file=temporary, end='')

    os.remove(config_path)
    with open(config_temporary_path, 'r+') as temporary:
        with open(config_path, 'w+') as config:
            for line in temporary:
                print(line, file=config, end='')
    os.remove(config_temporary_path)
