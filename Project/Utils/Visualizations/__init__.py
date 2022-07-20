"""
__init__.py of Visualizations
"""

import os

GENERATED_FOLDER_PATH = os.path.join(os.path.dirname(__file__), 'Generated_Files')
PATH_VISUALIZATION_FOLDER_PATH = os.path.join(GENERATED_FOLDER_PATH, 'Plot_Path_Output')
ICONCLASS_MEASURE_VISUALIZATION_FOLDER_PATH = os.path.join(GENERATED_FOLDER_PATH, 'Iconclass_Measure_Output')

for path in [GENERATED_FOLDER_PATH, PATH_VISUALIZATION_FOLDER_PATH, ICONCLASS_MEASURE_VISUALIZATION_FOLDER_PATH]:
    if not os.path.exists(path):
        os.makedirs(path)
