"""
__init__.py of FilterCache
"""

import os

GENERATED_FOLDER_PATH = os.path.join(os.path.dirname(__file__), 'Generated_Dataframe_Files')

FILTER_CACHES_PATH = os.path.join(GENERATED_FOLDER_PATH, 'Filter_Caches')

for path in [GENERATED_FOLDER_PATH, FILTER_CACHES_PATH]:
    if not os.path.exists(path):
        os.makedirs(path)

