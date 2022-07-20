"""
__init__.py of Project
"""
import os

GENERATED_FOLDER_PATH = os.path.join(os.path.dirname(__file__), 'Generated_Files')
BOXES_DICT_PATH = os.path.join(GENERATED_FOLDER_PATH, 'Boxes_Dictionary')
CONTAINS_TREES_DICT_PATH = os.path.join(GENERATED_FOLDER_PATH, 'Contains_Trees')
CONTAINS_PERSONS_DICT_PATH = os.path.join(GENERATED_FOLDER_PATH, 'Contains_Persons')

for path in [GENERATED_FOLDER_PATH]:
    if not os.path.exists(path):
        os.makedirs(path)
