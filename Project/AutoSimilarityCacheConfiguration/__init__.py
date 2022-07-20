"""
__init__ of AutoSimilarityCacheConfiguration
"""

import os

GENERATED_FOLDER_PATH = os.path.join(os.path.dirname(__file__), 'Generated_Dataframe_Files')

WEIGHT_CACHE_PATH = os.path.join(GENERATED_FOLDER_PATH, 'Weight_Cache')
CREATORS_CACHE_PATH = os.path.join(GENERATED_FOLDER_PATH, 'Creators_Cache')
TITLES_CACHE_PATH = os.path.join(GENERATED_FOLDER_PATH, 'Titles_Cache')
ALL_LABELS_CACHE_PATH = os.path.join(GENERATED_FOLDER_PATH, 'All_Labels')
ENHANCED_IMAGES_PATH = os.path.join(GENERATED_FOLDER_PATH, 'Enhanced_Images')
BOUNDING_BOXES_PKL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Notebooks', 'Object_Detection',
                                       'cleaned_bounding_boxes.pkl')
TREE_ANNOTATIONS_PKL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Generated_Files',
                                         'Contains_Trees.pkl')
PERSON_ANNOTATIONS_PKL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Generated_Files',
                                           'Contains_Persons.pkl')

for path in [GENERATED_FOLDER_PATH]:
    if not os.path.exists(path):
        os.makedirs(path)
