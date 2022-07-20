"""
__init__.py of Caching
"""

import os

GENERATED_FOLDER_PATH = os.path.join(os.path.dirname(__file__), 'Generated_Dataframe_Files')
SIMILARITY_AND_RANK_CACHES_PATH = os.path.join(GENERATED_FOLDER_PATH, 'Similarity_And_Rank_Caches')
SIMILARITY_AND_RANK_CACHES_GENERATED_IDENTIFIERS_PATH = os.path.join(SIMILARITY_AND_RANK_CACHES_PATH,
                                                                     'Similarity_And_Rank_Caches_Generated_Identifiers')
ALL_VALUES_CHECKED = os.path.join(SIMILARITY_AND_RANK_CACHES_PATH, 'All_values_checked')
TAG_AVERAGE_SIMILARITY_CACHES_PATH = os.path.join(GENERATED_FOLDER_PATH, 'Tag_Average_Similarity_Caches')
TAG_VECTOR_CACHE_PATH = os.path.join(GENERATED_FOLDER_PATH, 'Tag_Vector_Cache')
ICONCLASS_CACHE_PATH = os.path.join(GENERATED_FOLDER_PATH, 'Iconclass_Cache')

IDENTIFIER_COLUMN_NAME = '__IDENTIFIER__'

for path in [GENERATED_FOLDER_PATH, SIMILARITY_AND_RANK_CACHES_PATH, TAG_AVERAGE_SIMILARITY_CACHES_PATH,
             TAG_VECTOR_CACHE_PATH]:
    if not os.path.exists(path):
        os.makedirs(path)
