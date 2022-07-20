"""
IconclassComparisonVisualization
"""
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate

from Project.AutoSimilarityCache.Interface import filter_and_optimize_data_tuples
from Project.Misc.Misc import crop_border_of_all_sides
from Project.Utils.IconclassCache.IconclassCache import IconclassCache
from Project.Utils.Misc.Misc import boolean_values_to_origin_container
from Project.Matching_and_Similarity_Tasks.Semantic_Path import generate_path
from Project.AutoSimilarityCacheConfiguration.DataAccess import DataAccess
from Project.Utils.Misc.Nlp import NLP
from Project.Utils.Misc.OriginContainer import OriginContainer
from Project.Utils.Visualizations import ICONCLASS_MEASURE_VISUALIZATION_FOLDER_PATH


def visualize_iconclass_development_for_path(start_identifier: str, end_identifier: str, path_length: int,
                                             title_tags, exp_tags, icon_tags: bool, not_icon_tags: bool,
                                             obj_tags: bool, at_level_or_higher: int, to_file: bool = False,
                                             prefix: str = '') -> None:
    """
    :param start_identifier: Identifier from which to start the path
    :param end_identifier: Identifier at which to end the path
    :param path_length: Length of the path to be generated
    :param title_tags: True if tags generated from titles should be used, False otherwise
    :param exp_tags: True if tags generated from expert tags should be used, False otherwise
    :param icon_tags: If True, then expert tags are used that are from the Iconclass system.
    This is meant to be used only when generating paths for the iconclass measure.
    :param not_icon_tags: If True, then expert tags are used that are not from the Iconclass system.
    This is meant to be used only when generating paths for the iconclass measure.
    :param obj_tags: Whether to consider tags from the origin "Obj".
    :param at_level_or_higher: This parameter controls the granularity. More specifically it is the level which
    should ideally be used to describe an artwork. If a category is not as deep down in the tree as specified by
    "at_level_or_higher", then the finest available granularity is used instead.
    :param to_file: This parameter specifies wheter the results should be saved or not
    :param prefix: Prefix of the filenames if the files are to be saved.
    :return: None
    """
    assert at_level_or_higher > 0, 'Parameter \'at_level_or_higher\' must be at least 1'

    ic: IconclassCache = IconclassCache.instance
    da: DataAccess = DataAccess.instance
    tags_of_identifiers: Dict[str, List[str]] = dict()
    all_category_names: List[str] = []
    headers: List[str] = []
    iconclass_tags_dictionary: Dict[str, Tuple[str, ...]] = dict()

    path = generate_path(start=start_identifier, end=end_identifier, intermediate_steps=path_length,
                         title_tags=title_tags, exp_tags=exp_tags, icon_tags=icon_tags,
                         not_icon_tags=not_icon_tags, obj_tags=obj_tags)

    for ind, p in enumerate(path):
        tags = da.get_iconclass_tags_from_identifier(p)
        iconclass_tags_dictionary[p] = tags
        current_categories = []
        for j in range(len(tags)):
            current = ic.text_to_category(tags[j])
            assert current is not None
            current_categories.append(current)
        current_categories = [ic.get_category_at_level_or_higher(c, at_level_or_higher) for c in
                              current_categories]
        current_categories = [ic.category_to_text(c) for c in current_categories]

        tags_of_identifiers[p] = current_categories
        for category in current_categories:
            # This way the tags stay ordered, which would not be the case if a set was used
            # (which would be more efficient)
            if category not in all_category_names:
                all_category_names.append(category)
        headers.append(p)

    # defining an order of categories:
    start_categories: List[str] = tags_of_identifiers[path[0]]
    end_categories: List[str] = tags_of_identifiers[path[len(path) - 1]]

    assert len(start_categories) > 0, \
        'Artwork which is chosen as starting point does not have iconclass tags associated to it'
    assert len(end_categories) > 0, \
        'Artwork which is chosen as end point does not have iconclass tags associated to it'

    nlp = NLP.instance.nlp

    category_similarities: List[Tuple[str, float]] = []
    for c in all_category_names:
        current_similarities: List[float] = []
        for s in start_categories:
            current_similarities.append(nlp(s).similarity(nlp(c)) * 1 / len(start_categories))
        for e in end_categories:
            current_similarities.append(nlp(e).similarity(nlp(c)) * -1 / len(end_categories))
        category_similarities.append((c, sum(current_similarities)))
    all_category_names = [c[0] for c in sorted(category_similarities, reverse=True, key=lambda x: x[1])]

    rows: List[List] = []
    for category in all_category_names:
        percentages: List[float] = []
        for p in path:
            number_of_occurrences = 0
            search_in = tags_of_identifiers[p]
            for s in search_in:
                if s == category:
                    number_of_occurrences += 1
            percentages.append(number_of_occurrences / max(len(search_in), 1))
        rows.append(percentages)
    label_tags = []
    for category in all_category_names:
        if len(category) > 25:
            label_tags.append(category[:23] + '...')
        else:
            label_tags.append(category)
    plt.figure(figsize=(len(path) + 25, int(len(all_category_names))))
    title = f'Aggregation to hierarchy level {at_level_or_higher} (or higher) using ' \
            f'{"title tags and " if title_tags and not_icon_tags else ""}' \
            f'{"all expert tags that are not iconclass tags" if not_icon_tags else ""}' \
            f'{"iconclass tags" if icon_tags else ""}' \
            f'{"title tags " if title_tags and not not_icon_tags else ""}{"and " if title_tags and exp_tags else ""}' \
            f'{"unfiltered expert tags" if exp_tags else ""}' \
            f'{" and " if (title_tags or exp_tags or icon_tags or not_icon_tags) and obj_tags else ""}' \
            f'{"object tags" if obj_tags else ""}'
    plt.title(title, fontsize=24)
    heatmap = sns.heatmap(rows)
    heatmap.set_xticklabels(path, size=18)
    heatmap.set_yticklabels(label_tags, size=18, rotation=0)


    save_to: str = os.path.join(
        ICONCLASS_MEASURE_VISUALIZATION_FOLDER_PATH,
        f'{prefix}{"_" if len(prefix) != 0 else ""}iconclass_measure_{"icon" if icon_tags else ""}'
        f'{"title" if title_tags else ""}{"_" if title_tags and (exp_tags or not_icon_tags) else ""}'
        f'{"not_icon" if not_icon_tags else ""}{"exp" if exp_tags else ""}_{at_level_or_higher}'
        f'{"_" if (title_tags or exp_tags or icon_tags or not_icon_tags) and obj_tags else ""}'
        f'{"obj"}')
    if to_file:
        plt.savefig(save_to + '.png')
        crop_border_of_all_sides(save_to + '.png')
    else:
        plt.show()

    output: str = ''
    if not to_file:
        output += '\n\n\n'
    output += f'{title}:\n'

    output += 'Full category names:\n'
    for c in all_category_names:
        output += f' {c}, '
    output = output[:-2]
    output += '\n\n'

    headers = ['Category', 'Tag'] + headers
    for p in range(len(all_category_names)):
        rows[p] = [ic.text_to_category(all_category_names[p]), all_category_names[p]] + rows[p]
    output += tabulate(rows, headers=headers) + '\n\n'

    rows: List[Tuple[str, Tuple[str, ...]]] = []
    for p in path:
        rows.append((p, iconclass_tags_dictionary[p]))
    output += tabulate(rows, headers=['Identifier', 'Expert Annotations']) + '\n\n'

    rows: List[Tuple[str, str]] = []
    for p in path:
        rows.append((p, da.get_title_for_identifier(p)))
    output += tabulate(rows, headers=['Identifier', 'Title']) + '\n\n'

    origin_container: OriginContainer = boolean_values_to_origin_container(
        des=False, title=title_tags, exp=exp_tags, icon=icon_tags, not_icon=not_icon_tags, obj_tags=obj_tags)
    rows: List[Tuple[str, str]] = []
    for p in path:
        rows.append((p, str([tt[0] for tt in filter_and_optimize_data_tuples(p, origin_container) if tt[1] == 'Obj'])))
    output += tabulate(rows, headers=['Identifier', 'Found Objects']) + '\n\n'

    rows: List[Tuple[str, Tuple[Tuple[any, str]]]] = []
    output += 'The following tags were considered in the matching process:'
    for p in path:
        rows.append((p, filter_and_optimize_data_tuples(
                identifier=p, origin_container=boolean_values_to_origin_container(
                    des=False, title=title_tags, exp=exp_tags, icon=icon_tags, not_icon=not_icon_tags, obj_tags=obj_tags))))
    output += tabulate(rows, headers=['Identifier', 'Considered Tags']) + '\n\n'

    if to_file:
        with open(save_to + '.txt', 'w+', encoding='UTF-8') as f:
            print(output, file=f, end='')
    else:
        print(output)
