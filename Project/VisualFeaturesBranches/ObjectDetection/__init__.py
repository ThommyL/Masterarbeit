"""
innit of ObjectDetection
"""

import os

GENERATED_FOLDER_PATH = os.path.join(os.path.dirname(__file__), 'Generated_Files')

MODEL_STATE_DICTIONARY_PATH = os.path.join(GENERATED_FOLDER_PATH, 'State_Dictionary')

for path in [GENERATED_FOLDER_PATH, MODEL_STATE_DICTIONARY_PATH]:
    if not os.path.exists(path):
        os.makedirs(path)
