"""
Visualization.py
"""

import os
from typing import Tuple

import matplotlib.figure
import matplotlib.image
import matplotlib.pyplot as plt
import numpy as np

from Project.AutoSimilarityCache.Interface import filter_and_optimize_data_tuples
from Project.AutoSimilarityCacheConfiguration.DataAccess import DataAccess
from Project.Matching_and_Similarity_Tasks.Semantic_Path import generate_path
from Project.Misc.Misc import crop_border_of_all_sides
from Project.Utils.Misc.Misc import boolean_values_to_origin_container
from Project.Utils.Misc.OriginContainer import OriginContainer
from Project.Utils.Visualizations import PATH_VISUALIZATION_FOLDER_PATH


def plot_path(
        start: str, end: str, intermediate_steps: int, max_col_length: int = 3, title_tags: bool = True,
        exp_tags: bool = True, icon_tags: bool = False, not_icon_tags: bool = False, obj_tags: bool = False,
        print_tags: bool = False, plot_call: bool = False, print_tags_to_console: bool = True,
        save_to_file: bool = False) -> None:
    """
    :param start: String that identifies the artwork from which to start the path
    :param end: String that identifies the artwork at which to end the path
    :param intermediate_steps: The amount of images that should be found between each start and end identifier
    :param title_tags: Whether to consider tags from the origin "Title".
    :param exp_tags: Whether to consider tags from the origin "Exp".
    :param icon_tags: If True, then expert tags are used that are from the Iconclass system.
    This is meant to be used only when generating paths for the iconclass measure.
    :param not_icon_tags: If True, then expert tags are used that are not from the Iconclass system.
    :param obj_tags: Whether to consider tags from the origin "Obj".
    This is meant to be used only when generating paths for the iconclass measure.
    :param max_col_length: Specifies after how many images a new row is started
    :param print_tags: If True the tags associated for each identifier are printed
    :param plot_call: Whether fig.show() should be called or not
    :param print_tags_to_console: If true tags are printed into the console
    :param save_to_file: If True a png and or txt file are written to a new folder in the folder 'Plot_Path_Output'
    under the project root (which will be automatically created if it does not exist)
    :return: None
    """
    identifiers = generate_path(start, end, intermediate_steps, title_tags=title_tags, exp_tags=exp_tags,
                                icon_tags=icon_tags, obj_tags=obj_tags)

    da: DataAccess = DataAccess.instance

    for i in identifiers:
        assert da.is_valid_identifier(i), 'Could not find requested identifier in cleaned data'
    assert 1 < max_col_length <= 10, 'Parameter \'col_length\' must be > 1 and <= 10.'
    assert print_tags in [True, False], 'Parameter \'print_tags\' must be boolean.'
    assert plot_call in [True, False], 'Parameter \'plot_call\' must be boolean.'
    assert print_tags_to_console in [True, False], 'Parameter \'print_tags_to_console\' must be boolean.'
    assert save_to_file in [True, False], 'Parameter \'save_to_file\' must be boolean.'

    save_to: str = PATH_VISUALIZATION_FOLDER_PATH
    if not os.path.exists(save_to):
        os.makedirs(save_to)
    output_number: int = 0
    for _, _, files in os.walk(save_to):
        for file in files:
            output_number = max(output_number, int(file.split('.')[0]))
    output_number += 1

    save_to: str = os.path.join(PATH_VISUALIZATION_FOLDER_PATH, str(output_number))

    origin_container: OriginContainer = boolean_values_to_origin_container(
        des=False, title=title_tags, exp=exp_tags, icon=icon_tags, not_icon=not_icon_tags, obj_tags=obj_tags)
    print_tags_from: Tuple[str, ...] = origin_container.to_origin_tuple()

    if print_tags:
        image_nr: int = 0

        message: str = ''

        for i in identifiers:
            message += f'{i} -> Title: {da.get_title_for_identifier(i)}, Artist:{da.get_creator_for_identifier(i)}\n'

        message += '\n\n'

        for i in identifiers:
            message += f'{image_nr}: TAGS FOR {i}\n'
            tuples = tuple(t for t in da.get_tag_tuples_from_identifier(
                identifier=i, origin_container=origin_container) if t[1] in print_tags_from)
            for t in tuples:
                message += f'{t[0]} ->({t[1]})<-    :::    '
            if len(tuples) != len(actual := filter_and_optimize_data_tuples(
                    identifier=i, origin_container=origin_container)):
                message += f'\nThe following were considered for the matching: {tuple(a[0] for a in actual)}'
            message += '\n\n'
            image_nr += 1
        if print_tags_to_console:
            print(message)
        if save_to_file:
            with open(save_to + '.txt', 'w+') as f:
                print(message, end='', file=f)
    fig: matplotlib.figure.Figure
    axes: np.array
    fig, axes = plt.subplots(nrows=int((len(identifiers) - 1) / max_col_length) + 1,
                             ncols=min(len(identifiers), max_col_length))
    fig.set_figheight(25)
    fig.set_figwidth(25)

    ar = axes.ravel()
    for i in range(len(identifiers)):
        identifier: str = identifiers[i]
        ax = ar[i]
        img: np.ndarray = da.get_matplotlib_image_from_identifier(identifier)
        ax.imshow(img)
        image_title: str
        if i == 0:
            image_title = 'Start'
        elif i == len(identifiers) - 1:
            image_title = 'End'
        else:
            image_title = str(i)
        ax.set_title(image_title, fontsize=50)
    for ax in ar:
        ax.axis('off')
    fig.tight_layout()
    if save_to_file:
        plt.savefig(save_to + '.png')
        crop_border_of_all_sides(save_to + '.png')
    if plot_call:
        fig.show()
