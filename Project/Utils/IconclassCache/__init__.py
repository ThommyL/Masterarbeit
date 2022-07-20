"""
__init__.py of IconclassCache
"""

import os

GENERATED_FOLDER_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                     'IconclassCache/Generated_Dataframe_Files')

ICONCLASS_CACHE_PATH = os.path.join(GENERATED_FOLDER_PATH, 'Iconclass_Cache')

ICONCLASS_CACHE_GENERATION_WAS_SUCCESSFUL_PKL = os.path.join(ICONCLASS_CACHE_PATH, 'iconclass_cache_generation_success')
ICONCLASS_CATEGORY_TO_NAME_PKL = os.path.join(ICONCLASS_CACHE_PATH, 'iconclass_category_to_name')
ICONCLASS_NUMBER_OF_TRIES_PKL = os.path.join(ICONCLASS_CACHE_PATH, 'iconclass_number_of_tries_categories')

for path in [GENERATED_FOLDER_PATH, ICONCLASS_CACHE_PATH]:
    if not os.path.exists(path):
        os.makedirs(path)
