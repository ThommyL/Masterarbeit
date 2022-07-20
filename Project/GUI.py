"""
Contrary to the rest of this master thesis, not much attention was paid to code quality, as time was scarce and is
better used to improve other parts of the project. Therefore this file remains in need of a refactor.
"""
import os.path
import pickle
import random
from tkinter import *
from typing import List, Optional

import PIL
import numpy as np
from PIL import ImageTk

from Project import BOXES_DICT_PATH, CONTAINS_TREES_DICT_PATH, CONTAINS_PERSONS_DICT_PATH
from Project.AutoSimilarityCache.Interface import filter_and_optimize_data_tuples
from Project.AutoSimilarityCacheConfiguration.DataAccess import DataAccess
from Project.Matching_and_Similarity_Tasks import TooFewSamplesLeftException, InputIdentifierWithoutTagsException
from Project.Matching_and_Similarity_Tasks.Fill_Spaces import fill_spaces
from Project.Matching_and_Similarity_Tasks.Semantic_Path import generate_path
from Project.Matching_and_Similarity_Tasks.Similarity import get_most_similar_artworks_and_their_tags
from Project.Utils.FilterCache.FilterCache import FilterCache
from Project.Utils.Misc.OriginContainer import OriginContainer
from Project.Utils.Misc.Singleton import Singleton


# noinspection PyMissingOrEmptyDocstring,DuplicatedCode

@Singleton
class GUIHandler:
    def __init__(self):
        self.__background_color = 'gainsboro'
        self.__foreground_color = 'dark slate gray'
        self.__text_entry_background_color = 'slate gray'
        self.__active_background_color = self.__foreground_color
        self.__active_foreground_color = self.__background_color
        self.__components = []
        self.__create_empty_window()
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(0, weight=1)
        self.__use_title_tags = True
        self.__use_exp_tags = True
        self.__use_obj_tags = True
        self.__selected_origins: OriginContainer = OriginContainer(('Title', 'Exp'))
        self.__filters = dict({'search_terms_above': dict(), 'search_terms_below': dict(), 'must_contain_artists': [],
                               'must_not_contain_artists': [], 'start_year': -np.inf, 'end_year': np.inf,
                               'minimum_number_of_tags': 0, 'maximum_number_of_tags': np.inf,
                               'minimum_sum_of_weights': 0, 'maximum_sum_of_weights': np.inf, 'must_be_title': ''})
        self.__filter_scroll_position_upper = 4
        self.__filter_scroll_position_lower = 0
        self.__filter_description_lines = []
        self.__filter_line_dict = dict()
        self.__filters_changed = False
        filter_cache: FilterCache = FilterCache.instance
        self.__number_of_artworks_left = filter_cache.nr_of_identifiers_left(origin_container=self.__selected_origins)
        self.__nr_of_artworks_left_var = StringVar()
        da: DataAccess = DataAccess.instance
        self.__nr_of_artworks_left_var.set(
            f"{self.__number_of_artworks_left} artworks are left from {len(da.get_ids())}")
        self.__show_similar_artwork_from = 0
        self.__show_similar_row_length = 4
        self.__show_similar_buttons = []
        self.__shown_artworks = filter_cache.get_filtered_identifiers(origin_container=self.__selected_origins)
        self.__show_similar_number_of_rows = 2
        self.__apply_filter_callback = None
        self.__reset_selection_button = None
        self.__semantic_path_selected_image_1 = None
        self.__semantic_path_selected_title_1 = None
        self.__semantic_path_selected_creator_1 = None
        self.__semantic_path_selected_identifier_1 = None
        self.__semantic_path_selected_image_2 = None
        self.__semantic_path_selected_title_2 = None
        self.__semantic_path_selected_creator_2 = None
        self.__semantic_path_selected_identifier_2 = None
        self.__semantic_path_selected_others = []
        self.__semantic_path = None
        self.__semantic_path_draw_calculate_button = True
        self.__selection_window = None
        self.__semantic_path_scroll_position = 0
        self.__intermediate_steps = 2
        self.__fill_spots_elements = []
        self.__fill_spots_identifiers = []
        self.__fill_spots_identifiers_scroll_positions = []
        self.__fill_spots_is_result = False
        self.__set_path_length_elements = []
        self.__fill_spots_calculate_button = None
        self.__fill_spots_result = None
        self.__fill_spots_scroll_position = 0
        self.__show_reset_button = False
        self.__semantic_path_calculate_button = None

        self.__x_from = 0
        self.__y_from = 0
        self.__x_to = 0
        self.__y_to = 0
        self.__boxes = []
        self.__box_labels = []
        self.__canvas = None
        self.__canvas_image = None
        self.__last_label = ""
        self.__exclude = []
        self.__width_ratio = None
        self.__height_ratio = None
        self.__current_index = -1

        self.__trees_dictionary_path = CONTAINS_TREES_DICT_PATH + '.pkl'
        self.__contains_trees = None
        if not os.path.exists(self.__trees_dictionary_path):
            with open(self.__trees_dictionary_path, 'wb+') as f:
                pickle.dump(dict(), f)
        if self.__contains_trees is None:
            with open(self.__trees_dictionary_path, 'rb') as f:
                self.__contains_trees = pickle.load(f)

        self.__persons_dictionary_path = CONTAINS_PERSONS_DICT_PATH + '.pkl'
        self.__contains_persons = None
        if not os.path.exists(self.__persons_dictionary_path):
            with open(self.__persons_dictionary_path, 'wb+') as f:
                pickle.dump(dict(), f)
        if self.__contains_persons is None:
            with open(self.__persons_dictionary_path, 'rb') as f:
                self.__contains_persons = pickle.load(f)

        self.__boxes_dictionary_path = BOXES_DICT_PATH + '.pkl'
        if not os.path.exists(self.__boxes_dictionary_path):
            with open(self.__boxes_dictionary_path, 'wb+') as f:
                pickle.dump(dict(), f)
        with open(self.__boxes_dictionary_path, 'rb') as f:
            self.__boxes_dictionary = pickle.load(f)

        self.__open_start_menu()

    def __reset_gui(self):
        for c in self.__components:
            c.destroy()
        self.__components = []
        self.__apply_filter_callback = None
        if self.__semantic_path_selected_image_1 is not None:
            self.__semantic_path_selected_image_1.destroy()
            self.__semantic_path_selected_title_1.destroy()
            self.__semantic_path_selected_creator_1.destroy()
        if self.__semantic_path_selected_image_2 is not None:
            self.__semantic_path_selected_image_2.destroy()
            self.__semantic_path_selected_title_2.destroy()
            self.__semantic_path_selected_creator_2.destroy()
        for e in self.__semantic_path_selected_others:
            e.destroy()
        self.__semantic_path_selected_others = []
        for e in self.__fill_spots_elements:
            e.destroy()
        self.__fill_spots_elements = []
        self.__fill_spots_identifiers = []
        self.__fill_spots_is_result = False
        for e in self.__set_path_length_elements:
            e.destroy()
        self.__set_path_length_elements = []
        if self.__fill_spots_calculate_button is not None:
            self.__fill_spots_calculate_button.destroy()
            self.__fill_spots_calculate_button = None
        self.__fill_spots_result = None
        self.__fill_spots_identifiers_scroll_positions = []
        self.__fill_spots_scroll_position = 0
        self.__semantic_path_draw_calculate_button = True
        if self.__semantic_path_calculate_button is not None:
            self.__semantic_path_calculate_button.destroy()
            self.__semantic_path_calculate_button = None
        self.__boxes = []
        self.__box_labels = []
        self.__canvas = None
        self.__canvas_image = None
        self.__last_label = ""
        self.__width_ratio = None
        self.__height_ratio = None

    def __toggle_use_title_tags(self):
        self.__use_title_tags = not self.__use_title_tags

    def __toggle_use_exp_tags(self):
        self.__use_exp_tags = not self.__use_exp_tags

    def __toggle_use_obj_tags(self):
        self.__use_obj_tags = not self.__use_obj_tags

    def __create_empty_window(self):
        self.root = Tk()
        self.root.resizable(False, False)
        self.root.config(background=self.__background_color)

    def __register_component(self, component):
        self.__components.append(component)

    def start(self):
        """
        Starts the GUI
        :return: None
        """
        self.root.mainloop()

    @staticmethod
    def __get_image(identifier: Optional[str], width: int, height: int):
        da: DataAccess = DataAccess.instance
        if identifier is None:
            image = PIL.Image.open('Questionmark.png')
        else:
            image = da.get_PIL_image_from_identifier(identifier=identifier)

        def get_background(dim_1, dim_2, dim_3):
            return np.ones((dim_1, dim_2, dim_3)) * 220

        def get_padded(to_pad, pad_width, pad_to):
            im_arr = np.array(to_pad)
            if pad_width:
                addition = 0
                if (pad_to - to_pad.width) % 2 != 0:
                    addition = 1
                half = int((pad_to - to_pad.width) / 2)
                return PIL.Image.fromarray(np.concatenate(
                    (get_background(to_pad.height, half, 3), im_arr, get_background(to_pad.height, half + addition, 3)),
                    axis=1).astype(np.uint8))
            else:
                addition = 0
                if (pad_to - to_pad.height) % 2 != 0:
                    addition = 1
                half = int((pad_to - to_pad.height) / 2)
                return PIL.Image.fromarray(np.concatenate(
                    (get_background(half, to_pad.width, 3), im_arr, get_background(half + addition, to_pad.width, 3)),
                    axis=0).astype(np.uint8))

        if image.width > width:
            image = image.resize((width, int(width * image.height / image.width)), PIL.Image.LANCZOS)
        if image.height > height:
            image = image.resize((int(height * image.width / image.height), height), PIL.Image.LANCZOS)
        if image.width < width:
            image = get_padded(image, True, width)
        if image.height < height:
            image = get_padded(image, False, height)
        return ImageTk.PhotoImage(image)

    @staticmethod
    def __get_string_with_max_line_length(input_str: str, max_line_length: int) -> str:
        result = ''
        for input_part in input_str.split('\n'):
            current_input = input_part.split(' ')
            temp_result = ''
            cur_line = ''

            for i in current_input:
                if len(cur_line) + len(i) < max_line_length:
                    cur_line += i + ' '
                else:
                    temp_result += cur_line[:-1] + '\n'
                    cur_line = i + ' '
                if len(i) > max_line_length:
                    while len(cur_line) >= max_line_length and cur_line[max_line_length] not in [',', ' ', '.', '!',
                                                                                                 '?']:
                        temp_result += cur_line[0:max_line_length] + '-\n'
                        cur_line = cur_line[max_line_length:]
                        if len(cur_line) <= max_line_length:
                            cur_line += ' '
            if cur_line == '\n' or len(cur_line.strip()) != 0:
                temp_result += cur_line

            result += temp_result

        return result

    def __get_amount_of_active_filters(self):
        result = 0
        if self.__filters['minimum_number_of_tags'] != 0:
            result += 1
        if self.__filters['maximum_number_of_tags'] != np.inf:
            result += 1
        if self.__filters['minimum_sum_of_weights'] != 0:
            result += 1
        if self.__filters['maximum_sum_of_weights'] != np.inf:
            result += 1
        for _ in self.__filters['search_terms_above'].keys():
            result += 1
        for _ in self.__filters['search_terms_below'].keys():
            result += 1
        if len(self.__filters['must_contain_artists']) > 0:
            result += 1
        if len(self.__filters['must_not_contain_artists']) > 0:
            result += 1
        if self.__filters['start_year'] != -np.inf:
            result += 1
        if self.__filters['end_year'] != np.inf:
            result += 1
        if self.__filters['must_be_title'] != '':
            result += 1
        return result

    def __filters_to_string(self):
        result: str = ''

        def get_filter_nr_generator():
            filter_number = 0
            while True:
                filter_number += 1
                yield str(filter_number)

        filter_nr_generator = get_filter_nr_generator()

        if self.__filters['minimum_number_of_tags'] != 0:
            result += f'{next(filter_nr_generator)}: The minimum amount of tags is: ' \
                      f'{self.__filters["minimum_number_of_tags"]}\n'
        if self.__filters['maximum_number_of_tags'] != np.inf:
            result += f'{next(filter_nr_generator)}: The maximum amount of tags is: ' \
                      f'{self.__filters["maximum_number_of_tags"]}\n'
        if self.__filters['minimum_sum_of_weights'] != 0:
            result += f'{next(filter_nr_generator)}: The minimum sum of weights of tags is: ' \
                      f'{self.__filters["minimum_sum_of_weights"]}\n'
        if self.__filters['maximum_sum_of_weights'] != np.inf:
            result += f'{next(filter_nr_generator)}: The maximum sum of weights of tags is: ' \
                      f'{self.__filters["maximum_sum_of_weights"]}\n'
        for k in self.__filters['search_terms_above'].keys():
            result += f'{next(filter_nr_generator)}: The result must contain at least one tag that has a similarity ' \
                      f'>= {self.__filters["search_terms_above"][k]} to "{k[0]}"\n'
        for k in self.__filters['search_terms_below'].keys():
            result += f'{next(filter_nr_generator)}: The result must not contain any tags that have a similarity >= ' \
                      f'{self.__filters["search_terms_below"][k]} to "{k[0]}"\n'
        if len(self.__filters['must_contain_artists']) > 0:
            result += f'{next(filter_nr_generator)}: The artwork must be created by one of the following artists: '
        for a in self.__filters['must_contain_artists']:
            result += f'{a}, '
        if len(self.__filters['must_contain_artists']) > 0:
            result = result[:-2]
            result += '\n'
        if len(self.__filters['must_not_contain_artists']) > 0:
            result += f'{next(filter_nr_generator)}: The artwork must not be created by one of the following artists: '
        for a in self.__filters['must_not_contain_artists']:
            result += f'{a}, '
        if len(self.__filters['must_not_contain_artists']) > 0:
            result = result[:-2]
            result += '\n'
        if self.__filters['start_year'] != -np.inf:
            result += f'{next(filter_nr_generator)}: The artwork must be created after {self.__filters["start_year"]}\n'
        if self.__filters['end_year'] != np.inf:
            result += f'{next(filter_nr_generator)}: The artwork must be created before than ' \
                      f'{self.__filters["end_year"]}\n'
        if self.__filters['must_be_title'] != '':
            result += f'The artwork must have the title {self.__filters["must_be_title"]}\n'
        return result

    def __draw_back_button(self, root, back_method, row, columnspan: int, title="Back"):
        home_button = Button(
            root, text=title, command=back_method, bg=self.__background_color,
            fg=self.__foreground_color, font=('Arial', 24, 'bold'),
            activebackground=self.__active_background_color, activeforeground=self.__active_foreground_color
        )
        home_button.grid(row=row, column=0, rowspan=1, columnspan=columnspan, sticky=EW)
        if root == self.root:
            self.__register_component(home_button)

    def __draw_sources_checkbutton_row(self, root, row_nr: int, positions, columnspans):
        use_title_tags_var = BooleanVar(root)
        use_title_tags_var.set(self.__use_title_tags)
        use_exp_tags_var = BooleanVar(root)
        use_exp_tags_var.set(self.__use_exp_tags)
        use_obj_tags_var = BooleanVar(root)
        use_obj_tags_var.set(self.__use_obj_tags)

        def toggle_use_title_tags():
            self.__toggle_use_title_tags()
            use_title_tags_var.set(self.__use_title_tags)

        def toggle_use_exp_tags():
            self.__toggle_use_exp_tags()
            use_exp_tags_var.set(self.__use_exp_tags)

        def toggle_use_obj_tags():
            self.__toggle_use_obj_tags()
            use_obj_tags_var.set(self.__use_obj_tags)

        label_texts = 'Consider title tags', 'Consider expert tags', 'Consider image tags'

        def call_back_method(to_call):
            to_call()
            self.__filters_changed = True

        methods = lambda m=toggle_use_title_tags: call_back_method(m), \
                  lambda m=toggle_use_exp_tags: call_back_method(m), lambda m=toggle_use_obj_tags: call_back_method(m)
        variables = use_title_tags_var, use_exp_tags_var, use_obj_tags_var

        assert len(label_texts) == len(methods) == len(variables) == len(positions) == len(columnspans)

        for label, call_back, var, column_ind, column_span in zip(label_texts, methods, variables, positions,
                                                                  columnspans):
            use_title_tags_label = Label(root, text=label, font=('Arial', 14, 'bold'), width=27,
                                         bg=self.__background_color, fg=self.__foreground_color)

            if root == self.root:
                self.__register_component(use_title_tags_label)
            use_title_tags_label.grid(row=row_nr, column=column_ind, sticky=W, columnspan=column_span)

            use_title_tags_checkbutton = Checkbutton(
                root, command=call_back, bg=self.__background_color, highlightcolor=self.__background_color,
                font=('Arial', 10), variable=var, selectcolor=self.__background_color,
                activeforeground=self.__foreground_color, foreground=self.__foreground_color,
                background=self.__background_color, activebackground=self.__background_color,
                highlightbackground=self.__background_color, disabledforeground=self.__background_color, padx=20)

            if root == self.root:
                self.__register_component(use_title_tags_checkbutton)
            use_title_tags_checkbutton.grid(row=row_nr, column=column_ind, sticky=W, columnspan=column_span)

        return use_title_tags_var, use_exp_tags_var, use_obj_tags_var

    def __open_error_window(self, message, from_window):
        error_window = Toplevel(from_window)
        error_window.title('Error')
        error_window.resizable(False, False)
        error_label = Label(error_window, text=message, font=('Arial', 14),
                            bg=self.__background_color, fg=self.__foreground_color, padx=10, justify=LEFT)
        error_label.grid(row=0, sticky=NSEW)

        def close_window():
            error_window.destroy()

        ok_button = Button(
            error_window, text=f"Ok", command=close_window,
            bg=self.__background_color,
            fg=self.__foreground_color, font=('Arial', 10), activebackground=self.__active_background_color,
            activeforeground=self.__active_foreground_color, padx=20)
        ok_button.grid(row=1, sticky=NSEW)

    def __open_select_similar_window(self, var, select_from, to_select, from_window):
        selection_window = Toplevel(from_window)
        selection_window.title(f'Select {to_select}')
        selection_window.resizable(False, False)
        selection_window.config(background=self.__background_color)

        def select_entity(e):
            selection_window.destroy()
            var.set(e)

        for ind, entity in enumerate(select_from[0:10]):
            label = Label(selection_window, text=f'{self.__get_string_with_max_line_length(entity, 50)}',
                          font=('Arial', 14), bg=self.__background_color, fg=self.__foreground_color, padx=10,
                          justify=LEFT)
            label.grid(row=ind, column=0, sticky=W)

            padding_cell = Label(selection_window, padx=4, bg=self.__background_color)
            padding_cell.grid(row=ind, column=1)

            select_button = Button(
                selection_window, text=f"Select {to_select}", command=lambda e=entity: select_entity(e),
                bg=self.__background_color,
                fg=self.__foreground_color, font=('Arial', 10), activebackground=self.__active_background_color,
                activeforeground=self.__active_foreground_color, padx=20)
            select_button.grid(row=ind, column=2, sticky=NSEW)

    def __open_set_filter_window(self, from_window):
        filter_window = Toplevel(from_window)
        filter_window.title('Set Filter Menu')
        filter_window.resizable(False, False)
        filter_window.config(background=self.__background_color)

        self.__filter_scroll_position_upper = 4
        self.__filter_scroll_position_lower = 0

        window_title_label = Label(
            filter_window, text="Set Filter Menu", bg=self.__background_color, fg=self.__foreground_color,
            font=('Arial', 24, 'bold'), pady=20,
            activebackground=self.__active_background_color, activeforeground=self.__active_foreground_color
        )
        window_title_label.grid(row=0, columnspan=10, sticky=NSEW)
        da: DataAccess = DataAccess.instance

        def delete_filter_by_index(at_ind):
            cur_ind = -1

            def delete_filter(ind, at):
                if ind >= at:
                    return True
                return False

            if self.__filters['minimum_number_of_tags'] != 0:
                cur_ind += 1
                if delete_filter(cur_ind, at_ind):
                    self.__filters['minimum_number_of_tags'] = 0
                    redraw_active_filters_description()
                    return
            if self.__filters['maximum_number_of_tags'] != np.inf:
                cur_ind += 1
                if delete_filter(cur_ind, at_ind):
                    self.__filters['maximum_number_of_tags'] = np.inf
                    redraw_active_filters_description()
                    return
            if self.__filters['minimum_sum_of_weights'] != 0:
                cur_ind += 1
                if delete_filter(cur_ind, at_ind):
                    self.__filters['minimum_sum_of_weights'] = 0
                    redraw_active_filters_description()
                    return
            if self.__filters['maximum_sum_of_weights'] != np.inf:
                cur_ind += 1
                if delete_filter(cur_ind, at_ind):
                    self.__filters['maximum_sum_of_weights'] = 0
                    redraw_active_filters_description()
                    return
            for k in self.__filters['search_terms_above'].keys():
                cur_ind += 1
                if delete_filter(cur_ind, at_ind):
                    self.__filters['search_terms_above'].pop(k)
                    redraw_active_filters_description()
                    return
            for k in self.__filters['search_terms_below'].keys():
                cur_ind += 1
                if delete_filter(cur_ind, at_ind):
                    self.__filters['search_terms_below'].pop(k)
                    redraw_active_filters_description()
                    return
            if len(self.__filters['must_contain_artists']) > 0:
                cur_ind += 1
                if delete_filter(cur_ind, at_ind):
                    self.__filters['must_contain_artists'] = []
                    redraw_active_filters_description()
                    return
            if len(self.__filters['must_not_contain_artists']) > 0:
                cur_ind += 1
                if delete_filter(cur_ind, at_ind):
                    self.__filters['must_not_contain_artists'] = []
                    redraw_active_filters_description()
                    return
            if self.__filters['start_year'] != -np.inf:
                cur_ind += 1
                if delete_filter(cur_ind, at_ind):
                    self.__filters['start_year'] = -np.inf
                    redraw_active_filters_description()
                    return
            if self.__filters['end_year'] != np.inf:
                cur_ind += 1
                if delete_filter(cur_ind, at_ind):
                    self.__filters['end_year'] = np.inf
                    redraw_active_filters_description()
                    return
            if self.__filters['must_be_title'] != '':
                cur_ind += 1
                if delete_filter(cur_ind, at_ind):
                    self.__filters['must_be_title'] = ''
                    redraw_active_filters_description()
            raise Exception('Could not be deleted')

        padding_row_1 = Label(filter_window, pady=1, bg=self.__background_color)
        padding_row_1.grid(row=1, column=0, columnspan=10)

        search_term_entry_hint = Label(filter_window, text="Enter text", font=('Arial', 10),
                                       bg=self.__background_color,
                                       fg=self.__foreground_color)
        search_term_entry_hint.grid(row=2, column=1)

        threshold_entry_hint = Label(filter_window, text="Enter a number between 0 and 1", font=('Arial', 10),
                                     bg=self.__background_color, fg=self.__foreground_color, padx=10)
        threshold_entry_hint.grid(row=2, column=3)

        search_term_label = Label(filter_window, text="Search term:", font=('Arial', 14), bg=self.__background_color,
                                  fg=self.__foreground_color, padx=10)
        search_term_label.grid(row=3, column=0, sticky=EW)

        search_term_var = StringVar(filter_window)

        search_term_entry = Entry(filter_window, font=('Arial', 14), bg=self.__text_entry_background_color,
                                  textvariable=search_term_var, fg=self.__background_color)
        search_term_entry.grid(row=3, column=1, sticky=EW)

        threshold_label_var = StringVar(filter_window)
        threshold_label_var.set('Minimum required Similarity')

        threshold_label = Label(filter_window, textvariable=threshold_label_var, font=('Arial', 14),
                                bg=self.__background_color, fg=self.__foreground_color, padx=10, width=35)
        threshold_label.grid(row=3, column=2, sticky=EW)

        threshold_checkbutton_var = BooleanVar()
        threshold_checkbutton_var.set(True)

        def invert_threshold():
            if threshold_checkbutton_var.get():
                threshold_label_var.set('Minimum required Similarity')
            else:
                threshold_label_var.set('Maximum required Similarity')

        invert_threshold_checkbotton = Checkbutton(
            filter_window, command=invert_threshold, bg=self.__background_color, highlightcolor=self.__background_color,
            font=('Arial', 10), variable=threshold_checkbutton_var, selectcolor=self.__background_color,
            activeforeground=self.__foreground_color, foreground=self.__foreground_color,
            background=self.__background_color, activebackground=self.__background_color,
            highlightbackground=self.__background_color, disabledforeground=self.__background_color, padx=20)
        invert_threshold_checkbotton.grid(row=3, column=2, sticky=E)

        threshold_var = StringVar(filter_window)
        threshold_var.set("0.3")
        threshold_entry = Entry(filter_window, font=('Arial', 14), bg=self.__text_entry_background_color,
                                textvariable=threshold_var, fg=self.__background_color)
        threshold_entry.grid(row=3, column=3, sticky=EW)

        use_title_tags_var, use_exp_tags_var, use_obj_tags_var = self.__draw_sources_checkbutton_row(
            filter_window, 19, [0, 2, 3], [2, 1, 2])

        filter_description_var = StringVar()

        def get_currently_selected_origins(title, exp, obj):
            origins: List[str] = []

            if not (title or exp or obj):
                self.__open_error_window('At least one origin must be selected', filter_window)
            if title:
                origins.append('Title')
            if exp:
                origins.append('Exp')
            if obj:
                origins.append('Obj')
            return OriginContainer(origins=tuple(origins))

        def refresh():
            if self.__filters_changed:
                title = use_title_tags_var.get()
                exp = use_exp_tags_var.get()
                obj = use_obj_tags_var.get()

                self.__selected_origins = get_currently_selected_origins(title, exp, obj)
                fc.reset_filters()
                fc.set_rule_must_be_similar_to_tags(self.__filters['search_terms_above'])
                fc.set_rule_must_not_be_similar_to_tags(self.__filters['search_terms_below'])
                fc.set_rule_must_be_created_after(self.__filters['start_year'])
                fc.set_rule_must_be_created_before(self.__filters['end_year'])
                fc.set_rule_min_length(self.__filters['minimum_number_of_tags'])
                fc.set_rule_max_length(self.__filters['maximum_number_of_tags'])
                fc.set_rule_min_sum_of_weights(self.__filters['minimum_sum_of_weights'])
                fc.set_rule_max_sum_of_weights(self.__filters['maximum_sum_of_weights'])
                fc.set_rule_must_be_created_by_one_of(self.__filters['must_contain_artists'])
                fc.set_rule_must_not_be_created_by_one_of(self.__filters['must_not_contain_artists'])
                fc.set_rule_must_be_title(self.__filters['must_be_title'])
                self.__filters_changed = False
                self.__number_of_artworks_left = fc.nr_of_identifiers_left(origin_container=self.__selected_origins)
                self.__apply_filter_callback()
            self.__nr_of_artworks_left_var.set(
                f"{self.__number_of_artworks_left} artworks are left from {len(da.get_ids())}")

        active_filters_var = StringVar()

        def set_active_filter_var():
            amount_of_active_filters = self.__get_amount_of_active_filters()
            active_filters_var.set(
                f'{amount_of_active_filters} active filters{":" if self.__get_amount_of_active_filters() > 0 else ""}')

        def scroll_active_filters_description():
            set_to = '\n'.join(self.__filter_description_lines[
                               self.__filter_scroll_position_lower:self.__filter_scroll_position_upper])
            for _ in range(self.__filter_scroll_position_upper - len(self.__filter_description_lines)):
                set_to += '\n'
            while len(set_to.split('\n')) < 4:
                set_to += '\n'
            set_to += '\n'
            filter_description_var.set(set_to)

            active_filter_value = Label(filter_window, textvariable=filter_description_var, font=('Arial', 14),
                                        bg=self.__background_color, fg=self.__foreground_color, padx=10, justify=LEFT)
            active_filter_value.grid(row=17, column=0, rowspan=2, columnspan=5, sticky=W)

        def redraw_active_filters_description():
            set_active_filter_var()
            lines = []
            self.__filter_line_dict = dict()

            for ind, l in enumerate([i for i in self.__filters_to_string().split('\n') if len(i.strip()) > 0]):
                for ll in self.__get_string_with_max_line_length(l, 80).split('\n'):
                    if len(ll.strip()) > 0:
                        lines.append(ll)
                self.__filter_line_dict[ind] = len(lines)

            self.__filter_description_lines = lines
            self.__filter_scroll_position_lower = 0
            self.__filter_scroll_position_upper = min(len(self.__filter_description_lines), 4)

            def scroll_up():
                if self.__filter_scroll_position_lower == 0:
                    return
                self.__filter_scroll_position_lower -= 1
                self.__filter_scroll_position_upper -= 1
                scroll_active_filters_description()

            scroll_up_button = Button(
                filter_window, text="Scroll up", command=scroll_up, bg=self.__background_color,
                fg=self.__foreground_color, font=('Arial', 10), activebackground=self.__active_background_color,
                activeforeground=self.__active_foreground_color, padx=20)
            scroll_up_button.grid(row=17, column=3, columnspan=1, sticky=NSEW)

            def scroll_down():
                if self.__filter_scroll_position_lower == len(self.__filter_description_lines) - 1:
                    return
                self.__filter_scroll_position_lower += 1
                self.__filter_scroll_position_upper += 1
                scroll_active_filters_description()

            scroll_down_button = Button(
                filter_window, text="Scroll down", command=scroll_down, bg=self.__background_color,
                fg=self.__foreground_color, font=('Arial', 10), activebackground=self.__active_background_color,
                activeforeground=self.__active_foreground_color, padx=20)
            scroll_down_button.grid(row=18, column=3, columnspan=1, sticky=NSEW)
            scroll_active_filters_description()

        def add_search_term():
            self.__filters_changed = True
            try:
                threshold = float(threshold_var.get())
                if 0 <= threshold <= 1:
                    if threshold_checkbutton_var.get():
                        self.__filters['search_terms_above'][(search_term_var.get(), 'Title')] = threshold
                    else:
                        self.__filters['search_terms_below'][(search_term_var.get(), 'Title')] = threshold
                else:
                    self.__open_error_window('Threshold must be larger or equal 0 and smaller or equal 1',
                                             filter_window)
            except ValueError:
                print('Error in add search term')
                self.__open_error_window('Input must be a floating point number', filter_window)

            threshold_var.set('')
            search_term_var.set('')
            redraw_active_filters_description()

        padding_cell_2 = Label(filter_window, padx=4, bg=self.__background_color)
        padding_cell_2.grid(row=3, column=4)

        add_search_term_button = Button(
            filter_window, text="Add Keyword", command=add_search_term, bg=self.__background_color,
            fg=self.__foreground_color, font=('Arial', 10),
            activebackground=self.__active_background_color, activeforeground=self.__active_foreground_color, padx=20)
        add_search_term_button.grid(row=3, column=5, sticky=EW)

        padding_cell_3 = Label(filter_window, padx=4, bg=self.__background_color)
        padding_cell_3.grid(row=3, column=6)

        search_by_artist_entry_hint = Label(filter_window, text="Enter the name of the artist", font=('Arial', 10),
                                            bg=self.__background_color, fg=self.__foreground_color)
        search_by_artist_entry_hint.grid(row=5, column=1)

        search_by_artist_label = Label(filter_window, text="Artist:", font=('Arial', 14),
                                       bg=self.__background_color, fg=self.__foreground_color, padx=10)
        search_by_artist_label.grid(row=6, column=0, sticky=EW)

        search_by_artist_var = StringVar()

        search_by_artist_entry = Entry(filter_window, font=('Arial', 14), bg=self.__text_entry_background_color,
                                       fg=self.__background_color, textvariable=search_by_artist_var)
        search_by_artist_entry.grid(row=6, column=1, sticky=EW)

        padding_cell_4 = Label(filter_window, padx=4, bg=self.__background_color)
        padding_cell_4.grid(row=6, column=4)

        invert_artist_var = BooleanVar()
        invert_artist_var.set(True)

        def invert_artist_selection():
            if invert_artist_var.get():
                artist_label_var.set('From artist')
            else:
                artist_label_var.set('Not from artist')

        artist_label_var = StringVar()
        artist_label_var.set('From artist')
        invert_artist_selection_label = Label(filter_window, textvariable=artist_label_var, font=('Arial', 14),
                                              bg=self.__background_color, fg=self.__foreground_color, padx=10,
                                              width=25)
        invert_artist_selection_label.grid(row=6, column=2, sticky=EW)

        invert_artist_selection_checkbutton = Checkbutton(
            filter_window, command=invert_artist_selection, bg=self.__background_color,
            highlightcolor=self.__background_color,
            font=('Arial', 10), variable=invert_artist_var, selectcolor=self.__background_color,
            activeforeground=self.__foreground_color, foreground=self.__foreground_color,
            background=self.__background_color, activebackground=self.__background_color,
            highlightbackground=self.__background_color, disabledforeground=self.__background_color, padx=20)
        invert_artist_selection_checkbutton.grid(row=6, column=2, sticky=E)

        padding_cell_5 = Label(filter_window, padx=6, bg=self.__background_color)
        padding_cell_5.grid(row=6, column=4)

        def add_artist():
            self.__filters_changed = True
            if (artist := search_by_artist_var.get()) not in da.get_all_creators():
                self.__open_error_window('Artist was not found in the dataset', filter_window)
            elif invert_artist_var.get():
                self.__filters['must_contain_artists'].append(artist)
            else:
                self.__filters['must_not_contain_artists'].append(artist)
            search_by_artist_var.set('')
            redraw_active_filters_description()

        def search_artist():
            self.__open_select_similar_window(
                search_by_artist_var, da.get_closest_creators(search_by_artist_var.get()), 'artist', filter_window)

        search_for_artist_button = Button(
            filter_window, text="Search Artist", command=search_artist, bg=self.__background_color,
            fg=self.__foreground_color, font=('Arial', 10),
            activebackground=self.__active_background_color, activeforeground=self.__active_foreground_color, padx=20)
        search_for_artist_button.grid(row=6, column=3, sticky=EW)

        padding_cell_6 = Label(filter_window, padx=6, bg=self.__background_color)
        padding_cell_6.grid(row=6, column=4)

        add_artist_button = Button(
            filter_window, text="Add Artist", command=add_artist, bg=self.__background_color,
            fg=self.__foreground_color, font=('Arial', 10),
            activebackground=self.__active_background_color, activeforeground=self.__active_foreground_color, padx=20
        )

        add_artist_button.grid(row=6, column=5, sticky=EW)

        title_var = StringVar()

        search_title_hint = Label(filter_window, text="Enter a specific title",
                                  font=('Arial', 10), bg=self.__background_color, fg=self.__foreground_color,
                                  justify=LEFT)
        search_title_hint.grid(row=7, column=1)

        search_title_label = Label(filter_window, text="Title:", font=('Arial', 14),
                                   bg=self.__background_color, fg=self.__foreground_color, padx=10)
        search_title_label.grid(row=8, column=0, sticky=EW)

        search_title_entry = Entry(filter_window, font=('Arial', 14), bg=self.__text_entry_background_color,
                                   fg=self.__background_color, textvariable=title_var)
        search_title_entry.grid(row=8, column=1, sticky=EW)

        def set_title():
            self.__filters_changed = True
            if title_var.get() in da.get_all_titles():
                self.__filters['must_be_title'] = title_var.get()
            else:
                self.__open_error_window('Title is not in dataset', filter_window)
            title_var.set('')
            redraw_active_filters_description()

        def search_title():
            self.__open_select_similar_window(title_var, da.get_closest_titles(title_var.get()), 'title', filter_window)

        search_for_title_button = Button(
            filter_window, text="Search Title", command=search_title, bg=self.__background_color,
            fg=self.__foreground_color, font=('Arial', 10),
            activebackground=self.__active_background_color, activeforeground=self.__active_foreground_color, padx=20)
        search_for_title_button.grid(row=8, column=3, sticky=EW)

        set_title_button = Button(
            filter_window, text="Set Title", command=set_title, bg=self.__background_color,
            fg=self.__foreground_color, font=('Arial', 10),
            activebackground=self.__active_background_color, activeforeground=self.__active_foreground_color, padx=20
        )

        set_title_button.grid(row=8, column=5, sticky=EW)

        start_year_entry_hint = Label(filter_window, text="Enter the start year", font=('Arial', 10),
                                      bg=self.__background_color, fg=self.__foreground_color,
                                      justify=LEFT)
        start_year_entry_hint.grid(row=9, column=1)

        end_year_hint = Label(filter_window, text="Enter the end year",
                              font=('Arial', 10), bg=self.__background_color, fg=self.__foreground_color,
                              justify=LEFT)
        end_year_hint.grid(row=9, column=3)

        start_year_label = Label(filter_window, text="Start year:", font=('Arial', 14),
                                 bg=self.__background_color, fg=self.__foreground_color, padx=10)
        start_year_label.grid(row=10, column=0, sticky=EW)

        start_year_var = StringVar(filter_window)

        start_year_entry = Entry(filter_window, font=('Arial', 14), bg=self.__text_entry_background_color,
                                 textvariable=start_year_var,
                                 fg=self.__background_color)
        start_year_entry.grid(row=10, column=1, sticky=EW)

        end_year_label = Label(filter_window, text="End year:", font=('Arial', 14),
                               bg=self.__background_color, fg=self.__foreground_color, padx=10)
        end_year_label.grid(row=10, column=2, sticky=EW)

        end_year_var = StringVar(filter_window)

        end_year_entry = Entry(filter_window, font=('Arial', 14), bg=self.__text_entry_background_color,
                               textvariable=end_year_var,
                               fg=self.__background_color)
        end_year_entry.grid(row=10, column=3, sticky=EW)

        def set_years():
            self.__filters_changed = True
            if (start_year := start_year_var.get()) != '':
                try:
                    self.__filters['start_year'] = int(start_year)
                except ValueError:
                    self.__open_error_window('Year must be an integer', filter_window)
                start_year_var.set('')
            if (end_year := end_year_var.get()) != '':
                try:
                    self.__filters['end_year'] = int(end_year)
                except ValueError:
                    self.__open_error_window('Year must be an integer', filter_window)
                end_year_var.set('')
            redraw_active_filters_description()

        set_year_button = Button(
            filter_window, text="Set Years", command=set_years, bg=self.__background_color,
            fg=self.__foreground_color, font=('Arial', 10),
            activebackground=self.__active_background_color, activeforeground=self.__active_foreground_color, padx=20
        )
        set_year_button.grid(row=10, column=5, sticky=EW)

        year_filtering_note = Label(filter_window,
                                    text="The year filtering is done by year estimation, see the document "
                                         "for more details.",
                                    font=('Arial', 10), bg=self.__background_color, fg=self.__foreground_color, padx=20)
        year_filtering_note.grid(row=11, column=0, columnspan=10, sticky=W)

        padding_row_2 = Label(filter_window, pady=8, bg=self.__background_color)
        padding_row_2.grid(row=12, column=0, columnspan=10)

        number_of_tags_or_sources_label_var = StringVar()
        tags_or_sources_var = BooleanVar()
        tags_or_sources_var.set(True)
        min_or_max_var = BooleanVar()
        min_or_max_var.set(True)

        def refresh_number_of_tags_or_sources_label_var():
            if tags_or_sources_var.get():
                if min_or_max_var.get():
                    number_of_tags_or_sources_label_var.set('Minimum number of tags')
                else:
                    number_of_tags_or_sources_label_var.set('Maximum number of tags')
            else:
                if min_or_max_var.get():
                    number_of_tags_or_sources_label_var.set('Minimum sum of weights')
                else:
                    number_of_tags_or_sources_label_var.set('Maximum sum of weights')
            redraw_active_filters_description()

        refresh_number_of_tags_or_sources_label_var()

        number_of_tags_or_sources_label = Label(filter_window, textvariable=number_of_tags_or_sources_label_var,
                                                font=('Arial', 14), bg=self.__background_color,
                                                fg=self.__foreground_color,
                                                padx=10)
        number_of_tags_or_sources_label.grid(row=13, column=0)

        number_of_tags_or_sources_entry_var = StringVar()
        number_of_tags_or_sources_entry = Entry(filter_window, font=('Arial', 14),
                                                bg=self.__text_entry_background_color,
                                                fg=self.__background_color,
                                                textvariable=number_of_tags_or_sources_entry_var)
        number_of_tags_or_sources_entry.grid(row=13, column=1)

        tags_or_sources_checkbutton = Checkbutton(
            filter_window, command=refresh_number_of_tags_or_sources_label_var, bg=self.__background_color,
            highlightcolor=self.__background_color, font=('Arial', 10), variable=tags_or_sources_var,
            selectcolor=self.__background_color, activeforeground=self.__foreground_color,
            foreground=self.__foreground_color,
            background=self.__background_color, activebackground=self.__background_color, text='Tags/Sources',
            highlightbackground=self.__background_color, disabledforeground=self.__background_color, padx=20)
        tags_or_sources_checkbutton.grid(row=13, column=2, sticky=W)

        tags_or_sources_checkbutton = Checkbutton(
            filter_window, command=refresh_number_of_tags_or_sources_label_var, bg=self.__background_color,
            highlightcolor=self.__background_color, font=('Arial', 10), variable=min_or_max_var,
            selectcolor=self.__background_color, activeforeground=self.__foreground_color,
            foreground=self.__foreground_color,
            background=self.__background_color, activebackground=self.__background_color, text='Min/Max',
            highlightbackground=self.__background_color, disabledforeground=self.__background_color, padx=20)
        tags_or_sources_checkbutton.grid(row=13, column=2)

        def add_tags_or_sources_filter():
            self.__filters_changed = True

            if tags_or_sources_var.get():
                try:
                    value = int(number_of_tags_or_sources_entry_var.get())
                    if value < 0:
                        self.__open_error_window('Value must be greater equal 0', filter_window)
                    else:
                        if min_or_max_var.get():
                            self.__filters['minimum_number_of_tags'] = value
                        else:
                            self.__filters['maximum_number_of_tags'] = value
                except ValueError:
                    self.__open_error_window('Value must be an integer', filter_window)
            else:
                try:
                    value = float(number_of_tags_or_sources_entry_var.get())
                    if value < 0:
                        self.__open_error_window('Value must be greater equal 0', filter_window)
                    else:
                        if min_or_max_var.get():
                            self.__filters['minimum_sum_of_weights'] = value
                        else:
                            self.__filters['maximum_sum_of_weights'] = value
                except ValueError:
                    self.__open_error_window('Value must be a floating point number', filter_window)
            number_of_tags_or_sources_entry_var.set('')
            redraw_active_filters_description()

        add_tags_or_sources_button = Button(
            filter_window, text="Add Filter", command=add_tags_or_sources_filter, bg=self.__background_color,
            fg=self.__foreground_color, font=('Arial', 10),
            activebackground=self.__active_background_color, activeforeground=self.__active_foreground_color, padx=20)
        add_tags_or_sources_button.grid(row=13, column=3, stick=EW)

        sum_of_weights_note = Label(filter_window, text="For labels extracted from text the sum of weights equals the "
                                                        "number of origins. E.g. Sum of weights > 3 means that tags "
                                                        "are extracted for example from the title (=1) and at least "
                                                        "two expert tags.\n"
                                                        "Note that therefore the result will be empty if only titles "
                                                        "are considered and sum of weights > 1 is used for filtering.",
                                    font=('Arial', 10), bg=self.__background_color, fg=self.__foreground_color, padx=10)
        sum_of_weights_note.grid(row=14, column=0, columnspan=10, sticky=W)

        padding_row_3 = Label(filter_window, pady=8, bg=self.__background_color)
        padding_row_3.grid(row=15, column=0, columnspan=10)

        active_filter_label = Label(filter_window, textvariable=active_filters_var, font=('Arial', 14, 'bold'),
                                    bg=self.__background_color, fg=self.__foreground_color, padx=10)
        active_filter_label.grid(row=16, column=0, sticky=W)

        def remove_topmost_filter():
            self.__filters_changed = True
            if self.__get_amount_of_active_filters() == 0:
                return
            for i in range(0, len(self.__filter_description_lines) + 2):
                if self.__filter_scroll_position_lower >= self.__filter_line_dict[i]:
                    continue
                delete_filter_by_index(i)
                break

        remove_filter_button = Button(
            filter_window, text="Remove topmost filter", command=remove_topmost_filter,
            bg=self.__background_color,
            fg=self.__foreground_color, font=('Arial', 10),
            activebackground=self.__active_background_color, activeforeground=self.__active_foreground_color, padx=20
        )

        remove_filter_button.grid(row=16, column=2, columnspan=2, sticky=EW)

        redraw_active_filters_description()

        padding_row_5 = Label(filter_window, pady=8, bg=self.__background_color)
        padding_row_5.grid(row=21, column=0, columnspan=10)

        refresh_button = Button(
            filter_window, text="Apply Filter", command=refresh, bg=self.__background_color,
            fg=self.__foreground_color, font=('Arial', 24, 'bold'),
            activebackground=self.__active_background_color, activeforeground=self.__active_foreground_color)

        refresh_button.grid(row=22, column=0, rowspan=1, columnspan=10, sticky=EW)

        padding_row_6 = Label(filter_window, pady=8, bg=self.__background_color)
        padding_row_6.grid(row=23, column=0, columnspan=10)

        artworks_left_label = Label(filter_window, textvariable=self.__nr_of_artworks_left_var,
                                    font=('Arial', 14, 'bold'), bg=self.__background_color, fg=self.__foreground_color,
                                    justify=LEFT, padx=10)

        artworks_left_label.grid(row=24, column=0, columnspan=10, sticky=EW)

        padding_row_7 = Label(filter_window, pady=8, bg=self.__background_color)
        padding_row_7.grid(row=25, column=0, columnspan=10)

        def back_button_method():
            refresh()
            filter_window.destroy()

        self.__draw_back_button(filter_window, back_button_method, 26, 10, "Apply and return")

    def __draw_filter_info(self, from_window, row):
        artworks_left_label = Label(from_window, textvariable=self.__nr_of_artworks_left_var,
                                    font=('Arial', 14, 'bold'), bg=self.__background_color, fg=self.__foreground_color,
                                    justify=LEFT, padx=10)
        if from_window == self.root:
            self.__register_component(artworks_left_label)
        artworks_left_label.grid(row=row, columnspan=10, sticky=EW)

        padding_cell_1 = Label(from_window, padx=4, bg=self.__background_color)
        if from_window == self.root:
            self.__register_component(padding_cell_1)
        padding_cell_1.grid(row=row + 1, column=0)

        padding_cell_2 = Label(from_window, padx=4, bg=self.__background_color)
        if from_window == self.root:
            self.__register_component(padding_cell_2)
        padding_cell_2.grid(row=row + 1, column=9)

        change_filters = Button(
            from_window, text="Adapt Filters", command=lambda: self.__open_set_filter_window(from_window),
            bg=self.__background_color, fg=self.__foreground_color, font=('Arial', 10),
            activebackground=self.__active_background_color, activeforeground=self.__active_foreground_color, padx=20
        )
        if from_window == self.root:
            self.__register_component(change_filters)
        change_filters.grid(row=row + 2, column=1, columnspan=8, sticky=EW)

        padding_row_1 = Label(from_window, pady=1, bg=self.__background_color)
        if from_window == self.root:
            self.__register_component(padding_row_1)
        padding_row_1.grid(row=row + 3, column=0, columnspan=10)

    def __open_show_most_similar_artworks_window(self, selection_call_back, from_window=None, on_filter_change=None):
        if from_window is None:
            from_window = self.root
        from_window.title('Search Artwork')
        if from_window == self.root:
            self.__reset_gui()
        window_title_label_var = StringVar()
        window_title_label_var.set('Select an Artwork to find the most similar Artworks')
        window_title_label = Label(
            from_window, textvariable=window_title_label_var, bg=self.__background_color, fg=self.__foreground_color,
            font=('Arial', 24, 'bold'), pady=20,
            activebackground=self.__active_background_color, activeforeground=self.__active_foreground_color
        )
        if from_window == self.root:
            self.__register_component(window_title_label)
        window_title_label.grid(row=0, columnspan=10, sticky=NSEW)

        self.__draw_filter_info(from_window, 1)

        click_image_hint_label = Label(from_window, text='Click images for more info and to select an artwork',
                                       font=('Arial', 14, 'bold'), bg=self.__background_color,
                                       fg=self.__foreground_color,
                                       justify=LEFT, padx=10)
        if from_window == self.root:
            self.__register_component(click_image_hint_label)
        click_image_hint_label.grid(row=4, column=0, columnspan=10)

        padding_row_2 = Label(from_window, pady=1, bg=self.__background_color)
        if from_window == self.root:
            self.__register_component(padding_row_2)
        padding_row_2.grid(row=5, column=0, columnspan=10)

        showing_var = StringVar()
        showing_var.set(
            f'Showing artwork {self.__show_similar_artwork_from} to '
            f'{self.__show_similar_artwork_from + self.__show_similar_row_length}')

        showing_label = Label(from_window, textvariable=showing_var,
                              font=('Arial', 14, 'bold'), bg=self.__background_color,
                              fg=self.__foreground_color,
                              justify=LEFT, padx=10)
        if from_window == self.root:
            self.__register_component(showing_label)
        showing_label.grid(row=6, column=0, columnspan=10, sticky=EW)

        def scroll_up():
            self.__show_similar_artwork_from -= self.__show_similar_row_length
            self.__show_similar_artwork_from = max(0, self.__show_similar_artwork_from)
            show_artworks()

        scroll_up_button = Button(
            from_window, text="Scroll up", command=scroll_up,
            bg=self.__background_color, fg=self.__foreground_color, font=('Arial', 10),
            activebackground=self.__active_background_color, activeforeground=self.__active_foreground_color, padx=20
        )

        if from_window == self.root:
            self.__register_component(scroll_up_button)
        scroll_up_button.grid(row=6, column=1, columnspan=8, sticky=EW)

        def show_artworks():
            filter_cache: FilterCache = FilterCache.instance
            if selection_call_back != show_most_similar:
                self.__shown_artworks = filter_cache.get_filtered_identifiers(self.__selected_origins)

            for b in self.__show_similar_buttons:
                b.destroy()
            if self.__show_reset_button:
                padding_row_3 = Label(from_window, pady=5, bg=self.__background_color)
                self.__register_component(padding_row_3)
                padding_row_3.grid(row=11, column=0, columnspan=10)

                self.__reset_selection_button = Button(
                    from_window, text="Reset Selection", command=reset_selection,
                    bg=self.__background_color, fg=self.__foreground_color, font=('Arial', 24),
                    activebackground=self.__active_background_color, activeforeground=self.__active_foreground_color,
                    padx=20
                )

                self.__register_component(self.__reset_selection_button)
                self.__reset_selection_button.grid(row=12, column=0, columnspan=10, sticky=EW)

            for i in range(self.__show_similar_number_of_rows):
                for ind, a in enumerate(
                        self.__shown_artworks[self.__show_similar_artwork_from + i * self.__show_similar_row_length:
                                              self.__show_similar_artwork_from + (i + 1) *
                                              self.__show_similar_row_length]):
                    photo = self.__get_image(a, 250, 250)
                    b = Button(
                        from_window,
                        command=lambda identifier=a: self.__open_artwork_info_window(identifier, from_window,
                                                                                     selection_call_back),
                        bg=self.__background_color, fg=self.__foreground_color, font=('Arial', 10),
                        activebackground=self.__active_background_color,
                        activeforeground=self.__active_foreground_color,
                        padx=20, compound=BOTTOM, image=photo, relief=FLAT
                    )
                    b.photo = photo

                    self.__register_component(b)
                    b.grid(row=7 + i, column=ind + 4, sticky=EW)
                    self.__show_similar_buttons.append(b)

        def reset_selection():
            filter_cache: FilterCache = FilterCache.instance
            self.__shown_artworks = filter_cache.get_filtered_identifiers(self.__selected_origins)
            self.__filter_scroll_position_lower = 0
            self.__filter_scroll_position_upper = 4
            if self.__reset_selection_button is not None:
                self.__reset_selection_button.destroy()
                self.__reset_selection_button = None
            self.__show_reset_button = False
            show_artworks()

        if on_filter_change is None:
            self.__apply_filter_callback = reset_selection
        else:
            def on_set():
                reset_selection()
                on_filter_change()

            self.__apply_filter_callback = on_set

        def show_most_similar(similar_to):
            self.__shown_artworks = tuple(get_most_similar_artworks_and_their_tags(similar_to, self.__selected_origins,
                                                                                   100)[1].keys())
            self.__show_reset_button = True
            show_artworks()

        if selection_call_back is None:
            selection_call_back = show_most_similar

        padding_row_4 = Label(from_window, pady=5, bg=self.__background_color)
        if from_window == self.root:
            self.__register_component(padding_row_4)
        padding_row_4.grid(row=13, column=0, columnspan=10)

        show_artworks()

        def scroll_down():
            self.__show_similar_artwork_from += self.__show_similar_row_length

            if len(self.__shown_artworks) % self.__show_similar_row_length == 0:
                correction = 1
            else:
                correction = 0
            if int(self.__show_similar_artwork_from / self.__show_similar_row_length) >= int(
                    len(self.__shown_artworks) / self.__show_similar_row_length) - (
                    self.__show_similar_number_of_rows - 1) + correction:
                self.__show_similar_artwork_from = (int(len(self.__shown_artworks) / self.__show_similar_row_length) - (
                        self.__show_similar_number_of_rows - 1) + correction) * self.__show_similar_row_length
            show_artworks()

        scroll_down_button = Button(
            from_window, text="Scroll down", command=scroll_down,
            bg=self.__background_color, fg=self.__foreground_color, font=('Arial', 10),
            activebackground=self.__active_background_color, activeforeground=self.__active_foreground_color, padx=20
        )

        self.__register_component(scroll_down_button)
        scroll_down_button.grid(row=10, column=1, columnspan=8, sticky=EW)

        if from_window == self.root:
            def back_temp():
                self.__open_start_menu()
        else:
            def back_temp():
                from_window.destroy()
        if selection_call_back == show_most_similar:
            def back():
                reset_selection()
                back_temp()
        else:
            back = back_temp

        self.__draw_back_button(from_window, back, 14, 10)

    def __open_fill_spots_window(self):
        self.root.title('Fill Spots')
        self.__reset_gui()
        window_title_label = Label(
            self.root, text="Fill Spots", bg=self.__background_color, fg=self.__foreground_color,
            font=('Arial', 24, 'bold'), pady=20,
            activebackground=self.__active_background_color, activeforeground=self.__active_foreground_color
        )
        self.__register_component(window_title_label)
        window_title_label.grid(row=0, columnspan=10, sticky=NSEW)

        self.__draw_filter_info(self.root, 1)

        image_hint_var = StringVar()
        image_hint_var.set('Click images or questionmarks for more info and to select an artwork')
        click_image_hint_label = Label(self.root,
                                       textvariable=image_hint_var,
                                       font=('Arial', 14, 'bold'), bg=self.__background_color,
                                       fg=self.__foreground_color, pady=10,
                                       justify=LEFT, padx=10)
        self.__register_component(click_image_hint_label)
        click_image_hint_label.grid(row=4, column=0, columnspan=10)

        def draw_path(row):
            if self.__fill_spots_calculate_button is not None:
                self.__fill_spots_calculate_button.destroy()
                self.__fill_spots_calculate_button = None
            if len(self.__fill_spots_identifiers) == 0:
                image_hint_var.set('Enter length of path')
                length_entry_var = StringVar()
                length_entry_var.set('5')
                length_entry = Entry(self.root, font=('Arial', 14), bg=self.__text_entry_background_color,
                                     textvariable=length_entry_var, fg=self.__background_color)
                length_entry.grid(row=row, column=1, columnspan=3)
                self.__set_path_length_elements.append(length_entry)

                def set_identifier_list(length):
                    try:
                        length = int(length)
                    except ValueError:
                        self.__open_error_window('Value must be an integer', self.root)
                        return
                    if length < 3:
                        self.__open_error_window('Length must be greater equal 3', self.root)
                        return
                    for _ in range(length):
                        self.__fill_spots_identifiers.append(None)
                        self.__fill_spots_identifiers_scroll_positions.append(0)
                    for elem in self.__set_path_length_elements:
                        elem.destroy()
                    self.__set_path_length_elements = []
                    draw_path(row)

                length_entry_button = Button(
                    self.root, text="Set Length", command=lambda: set_identifier_list(length_entry_var.get()),
                    bg=self.__background_color, fg=self.__foreground_color, font=('Arial', 10),
                    activebackground=self.__active_background_color, activeforeground=self.__active_foreground_color,
                    padx=20
                )
                length_entry_button.grid(row=row, column=4, columnspan=3)
                self.__set_path_length_elements.append(length_entry_button)
            elif self.__fill_spots_is_result:
                image_hint_var.set('Scroll through the results')
                for ind, e in enumerate(self.__fill_spots_identifiers[
                                        self.__fill_spots_scroll_position: self.__fill_spots_scroll_position + 4]):
                    draw_selected(ind * 2 + 1, e, row, ind + self.__fill_spots_scroll_position)
            else:
                if self.__fill_spots_identifiers[0] is None or \
                        self.__fill_spots_identifiers[len(self.__fill_spots_identifiers) - 1] is None:
                    image_hint_var.set('At least the first and last identifier must be selected.')
                else:
                    image_hint_var.set('Continue to select artworks or generate suggestions.')
                    draw_calculate_button(row + 3)
                for ind, e in enumerate(self.__fill_spots_identifiers[
                                        self.__fill_spots_scroll_position: self.__fill_spots_scroll_position + 4]):
                    draw_selected(ind * 2 + 1, e, row, ind + self.__fill_spots_scroll_position)

            if len(self.__fill_spots_identifiers) > 4:
                def scroll_left():
                    if self.__fill_spots_scroll_position == 0:
                        return
                    for element in self.__fill_spots_elements:
                        element.destroy()
                        self.__fill_spots_elements = []
                    self.__fill_spots_scroll_position -= 1
                    draw_path(row)

                scroll_left_button = Button(
                    self.root, text="<-", command=scroll_left,
                    bg=self.__background_color, fg=self.__foreground_color, font=('Arial', 14, 'bold'),
                    activebackground=self.__active_background_color, activeforeground=self.__active_foreground_color,
                    padx=20
                )
                self.__fill_spots_elements.append(scroll_left_button)
                scroll_left_button.grid(row=row + 1, column=0, sticky=NSEW)

                def scroll_right():
                    if self.__fill_spots_scroll_position == len(self.__fill_spots_identifiers) - 4:
                        return
                    for element in self.__fill_spots_elements:
                        element.destroy()
                        self.__fill_spots_elements = []
                    self.__fill_spots_scroll_position += 1
                    draw_path(row)

                scroll_left_button = Button(
                    self.root, text="->", command=scroll_right,
                    bg=self.__background_color, fg=self.__foreground_color, font=('Arial', 14, 'bold'),
                    activebackground=self.__active_background_color, activeforeground=self.__active_foreground_color,
                    padx=20
                )
                self.__fill_spots_elements.append(scroll_left_button)
                scroll_left_button.grid(row=row + 1, column=10, sticky=NSEW)

        def draw_calculate_button(row):
            padding_row_1 = Label(self.root, pady=5, bg=self.__background_color)
            self.__register_component(padding_row_1)
            padding_row_1.grid(row=row, column=0, columnspan=10)

            def button_method():
                try:
                    if self.__fill_spots_result is None:
                        self.__fill_spots_result = fill_spaces(tuple(self.__fill_spots_identifiers),
                                                               self.__selected_origins)
                        self.__fill_spots_is_result = True
                        draw_path(5)
                except TooFewSamplesLeftException as e:
                    self.__open_error_window(e.__str__(), self.root)
                except InputIdentifierWithoutTagsException as e:
                    self.__open_error_window(e.__str__(), self.root)

            self.__fill_spots_calculate_button = Button(
                self.root, text="Calculate Results", command=button_method,
                bg=self.__background_color, fg=self.__foreground_color, font=('Arial', 14, 'bold'),
                activebackground=self.__active_background_color, activeforeground=self.__active_foreground_color,
                padx=20
            )
            self.__fill_spots_calculate_button.grid(row=row + 1, column=1, columnspan=8, sticky=EW)

        def update(identifier, index):
            self.__fill_spots_identifiers[index] = identifier
            draw_path(5)

        def draw_selected(col, identifier, row, index):
            if self.__selection_window is not None:
                self.__selection_window.destroy()

            if self.__fill_spots_is_result or self.__fill_spots_identifiers[index] is not None:
                def button_method():
                    self.__open_artwork_info_window(identifier, self.root)
            else:
                def button_method():
                    self.__selection_window = Toplevel(self.root)
                    self.__selection_window.resizable(False, False)
                    self.__selection_window.config(background=self.__background_color)
                    self.__open_show_most_similar_artworks_window(
                        lambda ident=identifier: update(ident, index), self.__selection_window, lambda: draw_path(5))

            if self.__fill_spots_is_result and len(self.__fill_spots_result[index]) > 1:
                def scroll_up(ind):
                    if self.__fill_spots_identifiers_scroll_positions[ind] == len(self.__fill_spots_result[ind]) - 1:
                        return
                    self.__fill_spots_identifiers_scroll_positions[ind] += 1
                    draw_path(row)

                scroll_up_button = Button(
                    self.root, text="Up", command=lambda: scroll_up(index),
                    bg=self.__background_color, fg=self.__foreground_color, font=('Arial', 14, 'bold'),
                    activebackground=self.__active_background_color, activeforeground=self.__active_foreground_color,
                    padx=20
                )
                scroll_up_button.grid(row=row, column=col, columnspan=2, sticky=EW)
                self.__fill_spots_elements.append(scroll_up_button)
            if self.__fill_spots_is_result:
                identifier = self.__fill_spots_result[index][self.__fill_spots_identifiers_scroll_positions[index]][0]
            selected = self.__get_image(identifier, width=400, height=400)
            text = f'Position {index}'
            if self.__fill_spots_is_result and len(self.__fill_spots_result[index]) > 1:
                text += f'\nRank: {self.__fill_spots_identifiers_scroll_positions[index]}'
            artwork_image_button = Button(self.root, image=selected, text=text, padx=20, pady=20, relief=FLAT,
                                          command=button_method, bg=self.__background_color, font=('Arial', 10),
                                          fg=self.__foreground_color, justify=CENTER, compound=BOTTOM)
            artwork_image_button.photo = selected
            self.__fill_spots_elements.append(artwork_image_button)
            artwork_image_button.grid(row=row + 1, column=col, rowspan=1, columnspan=2, sticky=EW)

            if self.__fill_spots_is_result and len(self.__fill_spots_result[index]) > 1:
                def scroll_down(ind):
                    if self.__fill_spots_identifiers_scroll_positions[ind] == 0:
                        return
                    self.__fill_spots_identifiers_scroll_positions[ind] -= 1
                    draw_path(row)

                scroll_up_button = Button(
                    self.root, text="Down", command=lambda: scroll_down(index),
                    bg=self.__background_color, fg=self.__foreground_color, font=('Arial', 14, 'bold'),
                    activebackground=self.__active_background_color, activeforeground=self.__active_foreground_color,
                    padx=20
                )
                scroll_up_button.grid(row=row + 2, column=col, columnspan=2, sticky=EW)
                self.__fill_spots_elements.append(scroll_up_button)
            for i in self.__fill_spots_identifiers:
                if i is not None:
                    draw_reset_button(row + 5)
                    break

        def reset(only_results):
            if only_results:
                self.__fill_spots_identifiers_scroll_positions = []
                for _ in self.__fill_spots_identifiers:
                    self.__fill_spots_identifiers_scroll_positions.append(0)
            else:
                self.__fill_spots_identifiers = []
                self.__fill_spots_identifiers_scroll_positions = []
            for e in self.__fill_spots_elements:
                e.destroy()
            self.__fill_spots_elements = []
            self.__fill_spots_is_result = False
            for e in self.__set_path_length_elements:
                e.destroy()
            self.__set_path_length_elements = []
            if self.__fill_spots_calculate_button is not None:
                self.__fill_spots_calculate_button.destroy()
                self.__fill_spots_calculate_button = None
            self.__fill_spots_result = None
            self.__fill_spots_scroll_position = 0
            draw_path(5)

        self.__apply_filter_callback = lambda: reset(True)

        def draw_reset_button(row):
            padding_row_1 = Label(self.root, pady=5, bg=self.__background_color)
            padding_row_1.grid(row=row, column=0, columnspan=10)
            self.__fill_spots_elements.append(padding_row_1)

            reset_button = Button(
                self.root, text="Reset", command=lambda: reset(False),
                bg=self.__background_color, fg=self.__foreground_color, font=('Arial', 14, 'bold'),
                activebackground=self.__active_background_color, activeforeground=self.__active_foreground_color,
                padx=20
            )
            self.__fill_spots_elements.append(reset_button)
            reset_button.grid(row=row + 1, column=1, columnspan=9, sticky=EW)

        padding_row_2 = Label(self.root, pady=5, bg=self.__background_color)
        self.__register_component(padding_row_2)
        padding_row_2.grid(row=12, column=0, columnspan=10)

        self.__draw_back_button(self.root, self.__open_start_menu, row=13, columnspan=11)

        draw_path(5)

    def __open_artwork_info_window(self, identifier, from_window, select_call_back=None):
        artwork_info_window = Toplevel(from_window)
        artwork_info_window.title(f'Artwork Information, identifier: {identifier}')
        artwork_info_window.resizable(False, False)
        artwork_info_window.config(background=self.__background_color)
        da: DataAccess = DataAccess.instance
        self.__draw_back_button(artwork_info_window, artwork_info_window.destroy, 0, 10)
        q = self.__get_image(identifier=identifier, width=400, height=400)
        photo_label = Label(artwork_info_window, image=q, padx=20, pady=20,
                            bg=self.__background_color, fg=self.__foreground_color, justify=CENTER)
        photo_label.photo = q
        photo_label.grid(row=1, column=0, rowspan=13, columnspan=1, sticky=W)

        padding_cell = Label(artwork_info_window, padx=10, bg=self.__background_color)
        padding_cell.grid(row=1, column=1)

        padding_row_1 = Label(artwork_info_window, pady=5, bg=self.__background_color)
        padding_row_1.grid(row=2, column=1, columnspan=9)

        title_label = Label(artwork_info_window, text="Title:", font=('Arial', 14), bg=self.__background_color,
                            fg=self.__text_entry_background_color)
        title_label.grid(row=2, column=2, sticky=W)

        title_value_label = Label(
            artwork_info_window, text=self.__get_string_with_max_line_length(
                da.get_title_for_identifier(identifier=identifier), 50), font=('Arial', 24), bg=self.__background_color,
            fg=self.__foreground_color)
        title_value_label.grid(row=3, column=2, sticky=W)

        creator_label = Label(artwork_info_window, text="Creator:", font=('Arial', 14), bg=self.__background_color,
                              fg=self.__text_entry_background_color)
        creator_label.grid(row=4, column=2, sticky=W)

        creator_value_label = Label(
            artwork_info_window, text=self.__get_string_with_max_line_length(
                da.get_creator_for_identifier(identifier=identifier), 50), font=('Arial', 24),
            bg=self.__background_color, fg=self.__foreground_color)
        creator_value_label.grid(row=5, column=2, sticky=W)

        year_label = Label(artwork_info_window, text="Year:", font=('Arial', 14), bg=self.__background_color,
                           fg=self.__text_entry_background_color)
        year_label.grid(row=6, column=2, sticky=W)

        year_value_label = Label(
            artwork_info_window, text=self.__get_string_with_max_line_length(
                da.get_creation_date_from_identifier(identifier=identifier), 50), font=('Arial', 24),
            bg=self.__background_color, fg=self.__foreground_color)
        year_value_label.grid(row=7, column=2, sticky=W)

        material_technique_label = Label(artwork_info_window, text="Material Technique:",
                                         font=('Arial', 14), bg=self.__background_color,
                                         fg=self.__text_entry_background_color)
        material_technique_label.grid(row=8, column=2, sticky=W)

        material_technique_value_label = Label(
            artwork_info_window, text=self.__get_string_with_max_line_length(
                da.get_material_technique_from_identifier(identifier=identifier), 50), font=('Arial', 24),
            bg=self.__background_color, fg=self.__foreground_color)
        material_technique_value_label.grid(row=9, column=2, sticky=W)

        object_class_label = Label(artwork_info_window, text="Object Class:",
                                   font=('Arial', 14), bg=self.__background_color,
                                   fg=self.__text_entry_background_color)
        object_class_label.grid(row=10, column=2, sticky=W)

        object_class_value_label = Label(
            artwork_info_window, text=self.__get_string_with_max_line_length(
                da.get_object_class_from_identifier(identifier=identifier), 50), font=('Arial', 24),
            bg=self.__background_color, fg=self.__foreground_color)
        object_class_value_label.grid(row=11, column=2, sticky=W)

        temporal_label = Label(artwork_info_window, text="Epoch:",
                               font=('Arial', 14), bg=self.__background_color,
                               fg=self.__text_entry_background_color)
        temporal_label.grid(row=12, column=2, sticky=W)

        temporal_value_label = Label(
            artwork_info_window, text=self.__get_string_with_max_line_length(
                da.get_temporal_from_identifier(identifier=identifier), 50), font=('Arial', 24),
            bg=self.__background_color, fg=self.__foreground_color)
        temporal_value_label.grid(row=13, column=2, sticky=W)

        padding_row_2 = Label(artwork_info_window, bg=self.__background_color)
        padding_row_2.grid(row=14, column=0, columnspan=10, sticky=NSEW)

        exp_tags_label = Label(artwork_info_window, text="Expert Annotation:", font=('Arial', 14),
                               bg=self.__background_color, fg=self.__text_entry_background_color)
        exp_tags_label.grid(row=15, column=0, sticky=W)

        exp_tags = da.get_expert_tags_from_identifier(identifier=identifier)
        exp_tags_string = ''
        for t in exp_tags:
            exp_tags_string += t + '; '
        exp_tags_string = exp_tags_string[:-2]

        exp_tags_value_label = Label(artwork_info_window,
                                     text=self.__get_string_with_max_line_length(exp_tags_string, 80),
                                     font=('Arial', 14), bg=self.__background_color, justify=LEFT,
                                     fg=self.__foreground_color)
        exp_tags_value_label.grid(row=16, column=0, sticky=W)

        title_tags_label = Label(artwork_info_window, text="Extracted tags from title:", font=('Arial', 14),
                                 bg=self.__background_color, fg=self.__text_entry_background_color)
        title_tags_label.grid(row=17, column=0, sticky=W)

        title_tags = [tt[0] for tt in da.get_tag_tuples_from_identifier(identifier=identifier,
                                                                        origin_container=OriginContainer(('Title',)))]
        title_tags_string = ''
        for t in title_tags:
            title_tags_string += t + '; '
        title_tags_string = title_tags_string[:-2]

        title_tags_value_label = Label(artwork_info_window,
                                       text=self.__get_string_with_max_line_length(title_tags_string, 80),
                                       font=('Arial', 14), bg=self.__background_color, justify=LEFT,
                                       fg=self.__foreground_color)
        title_tags_value_label.grid(row=18, column=0, sticky=W)

        expert_label = Label(artwork_info_window, text="Extracted tags from expert annotation:",
                             font=('Arial', 14), bg=self.__background_color, fg=self.__text_entry_background_color)
        expert_label.grid(row=19, column=0, sticky=W)

        exp_tags = [tt[0] for tt in da.get_tag_tuples_from_identifier(identifier=identifier,
                                                                      origin_container=OriginContainer(('Exp',)))]
        exp_tags_string = ''
        for t in exp_tags:
            exp_tags_string += t + '; '
        exp_tags_string = exp_tags_string[:-2]

        exp_tags_value_label = Label(artwork_info_window,
                                     text=self.__get_string_with_max_line_length(exp_tags_string, 80),
                                     font=('Arial', 14), bg=self.__background_color,
                                     fg=self.__foreground_color, justify=LEFT)
        exp_tags_value_label.grid(row=20, column=0, sticky=W)

        obj_label = Label(artwork_info_window, text="(Unique) extracted objects from image:",
                          font=('Arial', 14), bg=self.__background_color, fg=self.__text_entry_background_color)
        obj_label.grid(row=21, column=0, sticky=W)

        obj_tags = [tt[0] for tt in filter_and_optimize_data_tuples(identifier=identifier,
                                                                      origin_container=OriginContainer(('Obj',)))]
        obj_tags_string = ''
        for t in obj_tags:
            obj_tags_string += t + '; '
        obj_tags_string = obj_tags_string[:-2]

        obj_tags_value_label = Label(artwork_info_window,
                                     text=self.__get_string_with_max_line_length(obj_tags_string, 80),
                                     font=('Arial', 14), bg=self.__background_color,
                                     fg=self.__foreground_color, justify=LEFT)
        obj_tags_value_label.grid(row=22, column=0, sticky=W)

        padding_row_3 = Label(artwork_info_window, bg=self.__background_color)
        padding_row_3.grid(row=23, column=0, columnspan=10, sticky=EW)

        if (desc := da.get_description_from_identifier(identifier)) != '':
            show_description_button = Button(
                artwork_info_window, text="Show Description",
                command=lambda: self.__open_description_window(desc, artwork_info_window),
                bg=self.__background_color, fg=self.__foreground_color, font=('Arial', 10),
                activebackground=self.__active_background_color, activeforeground=self.__active_foreground_color,
                padx=20
            )
            show_description_button.grid(row=24, columnspan=10, sticky=EW)

        if select_call_back is not None:
            Button(
                artwork_info_window, text="Select", command=lambda: select_call_back(identifier),
                bg=self.__background_color,
                fg=self.__foreground_color, font=('Arial', 24, 'bold'),
                activebackground=self.__active_background_color, activeforeground=self.__active_foreground_color,
                padx=20
            ).grid(row=25, columnspan=10, sticky=EW)

    def __open_description_window(self, text, from_window):
        artwork_description_window = Toplevel(from_window)
        artwork_description_window.title('Artwork description')
        artwork_description_window.resizable(False, False)
        artwork_description_window.config(background=self.__background_color)
        artwork_description_label = Label(artwork_description_window,
                                          text=self.__get_string_with_max_line_length(text, 150),
                                          font=('Arial', 14), bg=self.__background_color,
                                          fg=self.__foreground_color, justify=LEFT)
        artwork_description_label.grid(row=0, column=0, sticky=W)

        Button(
            artwork_description_window, text="Close", command=artwork_description_window.destroy,
            bg=self.__background_color,
            fg=self.__foreground_color, font=('Arial', 10),
            activebackground=self.__active_background_color, activeforeground=self.__active_foreground_color, padx=20
        ).grid(row=1, sticky=EW)

    def __open_artwork_selection_by_identifier_window(self):
        self.root.title('Artwork Selection')
        self.__reset_gui()
        self.__draw_back_button(self.root, self.__open_start_menu, 0, 1)
        identifier_label = Label(self.root, text="Image Identifier:", font=('Arial', 14),
                                 bg=self.__background_color, fg=self.__foreground_color, padx=10)
        self.__register_component(identifier_label)
        identifier_label.grid(row=1, sticky=NSEW)

        value = StringVar(self.root)

        def continue_to_info():
            self.__open_artwork_info_window(value.get(), self.root)

        identifier_entry = Entry(self.root, font=('Arial', 14), bg=self.__text_entry_background_color,
                                 fg=self.__background_color, textvariable=value)
        self.__register_component(identifier_entry)
        identifier_entry.grid(row=2, sticky=NSEW)

        continue_button = Button(
            self.root, text="Continue", command=continue_to_info, bg=self.__background_color,
            fg=self.__foreground_color, font=('Arial', 10),
            activebackground=self.__active_background_color, activeforeground=self.__active_foreground_color, padx=20
        )

        self.__register_component(continue_button)
        continue_button.grid(row=3, sticky=NSEW)

    def __open_semantic_path_window(self):
        self.root.title('Draw semantic Path')
        self.__reset_gui()
        window_title_label_var = StringVar()
        window_title_label_var.set('Select two Artworks to generate a semantic Path')
        window_title_label = Label(
            self.root, textvariable=window_title_label_var, bg=self.__background_color, fg=self.__foreground_color,
            font=('Arial', 24, 'bold'), pady=20,
            activebackground=self.__active_background_color, activeforeground=self.__active_foreground_color
        )
        self.__register_component(window_title_label)
        window_title_label.grid(row=0, columnspan=10, sticky=NSEW)

        self.__draw_filter_info(self.root, 1)

        click_image_hint_label = Label(self.root,
                                       text='Click images or questionmarks for more info and to select an artwork',
                                       font=('Arial', 14, 'bold'), bg=self.__background_color,
                                       fg=self.__foreground_color,
                                       justify=LEFT, padx=10)
        self.__register_component(click_image_hint_label)
        click_image_hint_label.grid(row=4, column=0, columnspan=10)

        def increase_path_length():
            self.__intermediate_steps += 1
            window_title_label_var.set(f'Showing Semantic Path of Length: {2 + self.__intermediate_steps}')
            update()

        def draw_intermediate_steps_buttons():
            increase_intermediate_steps_button = Button(
                self.root, text="Increase amount of steps", command=increase_path_length,
                bg=self.__background_color, fg=self.__foreground_color, font=('Arial', 10),
                activebackground=self.__active_background_color, activeforeground=self.__active_foreground_color,
                padx=20
            )
            increase_intermediate_steps_button.grid(row=5, column=5, columnspan=5, sticky=NSEW)

            def decrease_path_length():
                self.__intermediate_steps = max(1, self.__intermediate_steps - 1)
                window_title_label_var.set(f'Showing Semantic Path of Length: {2 + self.__intermediate_steps}')
                update()

            decrease_intermediate_steps_button = Button(
                self.root, text="Decrease amount of steps", command=decrease_path_length,
                bg=self.__background_color, fg=self.__foreground_color, font=('Arial', 10),
                activebackground=self.__active_background_color, activeforeground=self.__active_foreground_color,
                padx=20
            )

            decrease_intermediate_steps_button.grid(row=5, column=0, columnspan=5, sticky=NSEW)

            padding_row_1 = Label(self.root, pady=5, bg=self.__background_color)
            padding_row_1.grid(row=6, column=0, columnspan=10)

            self.__semantic_path_selected_others.append(increase_intermediate_steps_button)
            self.__semantic_path_selected_others.append(decrease_intermediate_steps_button)
            self.__semantic_path_selected_others.append(padding_row_1)

        def draw_reset_button():
            if self.__semantic_path_selected_identifier_1 is not None or \
                    self.__semantic_path_selected_identifier_2 is not None:
                reset_button = Button(
                    self.root, text="Reset Selection", command=reset_selection,
                    bg=self.__background_color, fg=self.__foreground_color, font=('Arial', 10),
                    activebackground=self.__active_background_color, activeforeground=self.__active_foreground_color,
                    padx=20
                )
                self.__register_component(reset_button)
                reset_button.grid(row=12, column=0, columnspan=10, sticky=EW)

        def draw_selected(col, identifier, row, is_start, is_end):
            if self.__selection_window is not None:
                self.__selection_window.destroy()

            da: DataAccess = DataAccess.instance

            if identifier is None:
                title = ''
                creator = ''
                if is_start or is_end:
                    def button_method():
                        self.__selection_window = Toplevel(self.root)
                        self.__selection_window.resizable(False, False)
                        self.__selection_window.config(background=self.__background_color)
                        self.__open_show_most_similar_artworks_window(
                            lambda i: draw_selected(col, i, row, is_start, is_end), self.__selection_window, update)
                else:
                    raise Exception('NONE')
            else:
                def button_method():
                    self.__open_artwork_info_window(identifier, self.root)

                title = 'Title: ' + da.get_title_for_identifier(identifier)
                creator = 'Creator: ' + da.get_creator_for_identifier(identifier)

            selected = self.__get_image(identifier, width=450, height=450)
            artwork_image_button = Button(self.root, image=selected, padx=20, pady=20, relief=FLAT,
                                          command=button_method, bg=self.__background_color,
                                          fg=self.__foreground_color, justify=CENTER)
            artwork_image_button.photo = selected

            artwork_title_label = Label(self.root, text=title, padx=20, pady=20,
                                        bg=self.__background_color, fg=self.__foreground_color, justify=CENTER)

            artwork_creator_label = Label(self.root, text=creator, padx=20, pady=20,
                                          bg=self.__background_color, fg=self.__foreground_color, justify=CENTER)

            if is_start:
                if self.__semantic_path_selected_image_1 is not None:
                    self.__semantic_path_selected_image_1.destroy()
                    self.__semantic_path_selected_title_1.destroy()
                    self.__semantic_path_selected_creator_1.destroy()
                    self.__semantic_path_selected_identifier_1 = None
                self.__semantic_path_selected_image_1 = artwork_image_button
                self.__semantic_path_selected_title_1 = artwork_title_label
                self.__semantic_path_selected_creator_1 = artwork_creator_label
                self.__semantic_path_selected_identifier_1 = identifier
            elif is_end:
                if self.__semantic_path_selected_image_2 is not None:
                    self.__semantic_path_selected_image_2.destroy()
                    self.__semantic_path_selected_title_2.destroy()
                    self.__semantic_path_selected_creator_2.destroy()
                    self.__semantic_path_selected_identifier_2 = None
                self.__semantic_path_selected_image_2 = artwork_image_button
                self.__semantic_path_selected_title_2 = artwork_title_label
                self.__semantic_path_selected_creator_2 = artwork_creator_label
                self.__semantic_path_selected_identifier_2 = identifier
            else:
                self.__semantic_path_selected_others.append(artwork_image_button)
                self.__semantic_path_selected_others.append(artwork_title_label)
                self.__semantic_path_selected_others.append(artwork_creator_label)

            artwork_image_button.grid(row=row, column=col, rowspan=1, columnspan=2, sticky=EW)
            artwork_title_label.grid(row=row + 1, column=col, rowspan=1, columnspan=2, sticky=EW)
            artwork_creator_label.grid(row=row + 2, column=col, rowspan=1, columnspan=2, sticky=EW)
            if self.__semantic_path is None:
                draw_path(row)
            draw_reset_button()

        def draw_path(row):
            def update_semantic_path():
                try:
                    self.__semantic_path = generate_path(self.__semantic_path_selected_identifier_1,
                                                         self.__semantic_path_selected_identifier_2,
                                                         self.__intermediate_steps, self.__use_title_tags,
                                                         self.__use_exp_tags)
                    self.__semantic_path_draw_calculate_button = False
                    draw_path(row)
                except TooFewSamplesLeftException as e:
                    self.__open_error_window(e.__str__(), self.root)
                    return
                except InputIdentifierWithoutTagsException as e:
                    self.__open_error_window(e.__str__(), self.root)
                    return

            if self.__semantic_path_calculate_button is not None:
                self.__semantic_path_calculate_button.destroy()
                self.__semantic_path_calculate_button = None
            if self.__semantic_path_selected_identifier_1 is None or \
                    self.__semantic_path_selected_identifier_2 is None:
                return
            if self.__semantic_path_draw_calculate_button:
                def calculate():
                    if self.__semantic_path_selected_identifier_1 == self.__semantic_path_selected_identifier_2:
                        self.__open_error_window('Artworks must be unique', self.root)
                        return
                    update_semantic_path()

                self.__semantic_path_calculate_button = Button(
                    self.root, text="Calculate", command=calculate,
                    bg=self.__background_color, fg=self.__foreground_color, font=('Arial', 10),
                    activebackground=self.__active_background_color,
                    activeforeground=self.__active_foreground_color, padx=20
                )
                self.__semantic_path_calculate_button.grid(row=row + 3, column=0, columnspan=10, sticky=NSEW)
            else:
                draw_intermediate_steps_buttons()

                if self.__intermediate_steps > 2:
                    def scroll_left():
                        if self.__semantic_path_scroll_position > 0:
                            self.__semantic_path_scroll_position -= 1
                            draw_path(7)

                    scroll_left_button = Button(
                        self.root, text="Scroll Left", command=scroll_left,
                        bg=self.__background_color, fg=self.__foreground_color, font=('Arial', 10),
                        activebackground=self.__active_background_color,
                        activeforeground=self.__active_foreground_color, padx=20
                    )
                    self.__register_component(scroll_left_button)
                    scroll_left_button.grid(row=row + 3, column=0, columnspan=5, sticky=NSEW)

            if self.__semantic_path is None:
                update_semantic_path()

            if self.__intermediate_steps == 1:
                draw_selected(1, self.__semantic_path[0], 7, False, False)
                draw_selected(4, self.__semantic_path[1], 7, False, False)
                draw_selected(7, self.__semantic_path[2], 7, False, False)
            else:
                for ind, p in enumerate(self.__semantic_path[
                                        self.__semantic_path_scroll_position:self.__semantic_path_scroll_position + 4]):
                    draw_selected(ind * 2 + 1, p, 7, False, False)

            if self.__intermediate_steps > 2:
                def scroll_right():
                    if self.__semantic_path_scroll_position < len(self.__semantic_path) - 4:
                        self.__semantic_path_scroll_position += 1
                        draw_path(7)

                scroll_right_button = Button(
                    self.root, text="Scroll Right", command=scroll_right,
                    bg=self.__background_color, fg=self.__foreground_color, font=('Arial', 10),
                    activebackground=self.__active_background_color,
                    activeforeground=self.__active_foreground_color, padx=20
                )
                self.__register_component(scroll_right_button)
                scroll_right_button.grid(row=row + 3, column=5, columnspan=5, sticky=NSEW)
                padding_row_2 = Label(self.root, pady=5, bg=self.__background_color)
                self.__register_component(padding_row_2)
                padding_row_2.grid(row=row + 4, column=0, columnspan=10)

        def reset_selection():
            if self.__semantic_path_selected_image_1 is not None:
                self.__semantic_path_selected_image_1.destroy()
                self.__semantic_path_selected_title_1.destroy()
                self.__semantic_path_selected_creator_1.destroy()
                self.__semantic_path_selected_identifier_1 = None
            if self.__semantic_path_selected_identifier_2 is not None:
                self.__semantic_path_selected_image_2.destroy()
                self.__semantic_path_selected_title_2.destroy()
                self.__semantic_path_selected_creator_2.destroy()
                self.__semantic_path_selected_identifier_2 = None
            for e in self.__semantic_path_selected_others:
                e.destroy()
            self.__semantic_path = None
            self.__semantic_path_scroll_position = 0
            self.__semantic_path_draw_calculate_button = True
            if self.__semantic_path_calculate_button is not None:
                self.__semantic_path_calculate_button.destroy()
                self.__semantic_path_calculate_button = None
            draw_selected(1, None, 7, True, False)
            draw_selected(5, None, 7, False, True)

        reset_selection()
        padding_row_3 = Label(self.root, pady=1, bg=self.__background_color)
        self.__register_component(padding_row_3)
        padding_row_3.grid(row=13, column=0, columnspan=10)
        self.__draw_back_button(self.root, self.__open_start_menu, 14, 10)

        def update():
            for e in self.__semantic_path_selected_others:
                e.destroy()
            self.__semantic_path = None
            self.__semantic_path_scroll_position = 0
            draw_path(7)

        self.__apply_filter_callback = update

    def __open_read_me(self):
        self.root.title('Info')
        self.__reset_gui()
        read_me_text = 'In every menu there is an option to set filters. These filters are applied globally, meaning ' \
                       'that if you set the filter e.g for searching a particular image and then generate a semantic ' \
                       'path for example, the semantic path too will only contain artworks that fullfill the ' \
                       'requirements of the filtering. It is always recommended to filter for a minimum number of ' \
                       'tags or a minimum sum of weights, as this greatly improves the results. Note also that in ' \
                       'the filter menu, the origins of tags that are taken into account can be changed. If you ' \
                       'unset "Consider Title Tags" for example, then tags extracted from the titles will not be ' \
                       'used in ANY of the processes.\n\n It often happens that one artwork is optimal for several ' \
                       'positions and in "Create Semantic Path", this is taken care of by maximizing a score. ' \
                       'However, in "Fill Spots" the aim is to show all the options in their best order for each and ' \
                       'every position on the path and therefore it will happen that suggestions for some positions ' \
                       'quite similar.\n\n This is all there is to know - Have fun experimenting!'
        read_me_label = Label(self.root, text=self.__get_string_with_max_line_length(read_me_text, 150), padx=20,
                              pady=20, bg=self.__background_color, fg=self.__foreground_color, justify=LEFT,
                              font=('Arial', 14))
        self.__register_component(read_me_label)
        read_me_label.grid(row=0, sticky=NSEW)
        self.__draw_back_button(self.root, self.__open_start_menu, 1, 1)

    def __open_start_menu(self):
        self.root.title('Home Menu')
        self.__reset_gui()
        texts = 'Search Artworks /\nShow most similar Artworks', 'Create Semantic Path', 'Fill Spots', \
                'Artwork Info', 'Please Read me first!', 'Annotation Window'
        methods = lambda: self.__open_show_most_similar_artworks_window(None), \
                  self.__open_semantic_path_window, self.__open_fill_spots_window, \
                  self.__open_artwork_selection_by_identifier_window, self.__open_read_me, self.__open_annotation_window
        colors = 'SpringGreen3', 'DodgerBlue3', 'RosyBrown3', 'MediumPurple3', 'plum3', 'RoyalBlue3'
        active_colors = 'SpringGreen4', 'DodgerBlue4', 'RosyBrown4', 'MediumPurple4', 'plum4', 'RoyalBlue4'

        assert len(texts) == len(methods) == len(colors) == len(active_colors)
        buttons = []
        for text, method, color, active_color, r in zip(
                texts, methods, colors, active_colors, range(1, len(texts) + 1)):
            buttons.append(
                b := Button(
                    self.root, text=text, command=method, bg=color, fg=active_color, font=('Arial', 50, 'bold'),
                    activebackground=active_color, activeforeground=color,
                    justify=CENTER, padx=2, pady=2
                )
            )

            b.grid(row=r, sticky=NSEW)
            self.__components.append(b)

    def __open_annotation_window(self, identifier=None, show_already_annotated=False):
        self.root.title('Image Annotation')
        self.__reset_gui()
        da: DataAccess = DataAccess.instance
        ids = list(da.get_ids())
        if not show_already_annotated:
            random.shuffle(ids)
        if identifier is None:
            for i in ids:
                if show_already_annotated:
                    if i in self.__boxes_dictionary.keys() and i not in self.__exclude:
                        identifier = i
                        break
                else:
                    if i not in self.__boxes_dictionary.keys() and i not in self.__exclude:
                        identifier = i
                        break
        if identifier is None:
            self.__open_start_menu()

        self.__canvas_image = da.get_PIL_image_from_identifier(identifier=identifier, enhanced=True)
        original_width = self.__canvas_image.width
        original_height = self.__canvas_image.height
        self.__canvas_image = self.__canvas_image.resize((800, 800), PIL.Image.LANCZOS)
        self.__width_ratio = original_width / self.__canvas_image.width
        self.__height_ratio = original_height / self.__canvas_image.height
        self.__canvas_image = ImageTk.PhotoImage(self.__canvas_image)

        self.__canvas = Canvas(self.root, bg=self.__background_color, height=self.__canvas_image.height(),
                               width=self.__canvas_image.width())
        self.__canvas.grid(row=3, column=0)
        self.__register_component(self.__canvas)

        self.__x_from = 0
        self.__y_from = 0
        self.__x_to = 0
        self.__y_to = 0

        if identifier in self.__boxes_dictionary.keys():
            self.__boxes = self.__boxes_dictionary[identifier][0]
            scaled = []
            for b in self.__boxes:
                scaled.append((b[0] / self.__width_ratio, b[1] / self.__height_ratio,
                               b[2] / self.__width_ratio, b[3] / self.__height_ratio))
            self.__boxes = scaled
            self.__box_labels = list(self.__boxes_dictionary[identifier][1])
        else:
            self.__boxes = []
            self.__box_labels = []

        def set_x_y(click):
            self.__x_from = click.x
            self.__y_from = click.y
            self.__x_to = 0
            self.__y_to = 0

        def within_bounds(val, is_width):
            if val < 0:
                return 0
            if is_width:
                if val > self.__canvas_image.width():
                    return self.__canvas_image.width()
            else:
                if val > self.__canvas_image.height():
                    return self.__canvas_image.height()
            return int(val)

        def box_within_bounds(box):
            return (within_bounds(box[0], True), within_bounds(box[1], False), within_bounds(box[2], True),
                    within_bounds(box[3], False))

        def order_box_coordinates(box):
            x_1, y_1, x_2, y_2 = box
            if x_1 > x_2:
                x_1, x_2 = x_2, x_1
            if y_1 > y_2:
                y_1, y_2 = y_2, y_1
            return x_1, y_1, x_2, y_2

        def fix_box(box):
            return order_box_coordinates(box_within_bounds(box))

        def draw_rectangle(x_1, y_1, x_2, y_2):
            x_1, y_1, x_2, y_2 = fix_box((x_1, y_1, x_2, y_2))
            self.__canvas.create_rectangle(x_1, y_1, x_2, y_2, width=1, fill="")
            self.__canvas.create_rectangle(x_1 + 1, y_1 + 1, x_2 - 1, y_2 - 1, width=1, fill="", outline='white')
            self.__canvas.create_rectangle(x_1 + 2, y_1 + 2, x_2 - 2, y_2 - 2, width=1, fill="")

        def set_rectangle(drag):
            self.__canvas.create_image(0, 0, anchor=NW, image=self.__canvas_image)
            self.__canvas.photo = self.__canvas_image
            self.__x_to = drag.x
            self.__y_to = drag.y
            draw_rectangle(self.__x_from, self.__y_from, self.__x_to, self.__y_to)

        def open_label_input_window(callback):
            label_input_window = Toplevel(self.root)
            label_input_window.title('Artwork description')
            label_input_window.resizable(False, False)
            label_input_window.config(background=self.__background_color)
            artwork_description_label = Label(label_input_window, text='Label:', font=('Arial', 14),
                                              bg=self.__background_color, fg=self.__foreground_color, justify=LEFT)
            artwork_description_label.grid(row=0, column=0, sticky=W)

            label_var = StringVar(label_input_window)

            label_entry = Entry(label_input_window, font=('Arial', 14), bg=self.__text_entry_background_color,
                                textvariable=label_var, fg=self.__background_color)
            label_entry.grid(row=2, column=1, sticky=EW)

            def on_return():
                label = label_var.get()
                if len(label) == 0:
                    if len(self.__last_label) != 0:
                        callback(self.__last_label)
                        label_input_window.destroy()
                else:
                    self.__last_label = label
                    callback(label)
                    label_input_window.destroy()

            label_entry.focus()
            label_entry.bind("<Return>", (lambda _: on_return()))

        def draw_boxes():
            self.__canvas.create_image(0, 0, anchor=NW, image=self.__canvas_image)
            self.__canvas.photo = self.__canvas_image
            for box, t in zip(self.__boxes, self.__box_labels):
                draw_rectangle(*box)
                self.__canvas.create_text(int((box[0] + box[2]) / 2), box[1] + 10, text=t)

        def register_square(_):
            def add_box_label(label):
                self.__boxes.append((self.__x_from, self.__y_from, self.__x_to, self.__y_to))
                self.__x_from = 0
                self.__y_from = 0
                self.__x_to = 0
                self.__y_to = 0
                self.__box_labels.append(label)
                draw_boxes()

            open_label_input_window(add_box_label)

        self.__canvas.bind('<Button-1>', set_x_y)
        self.__canvas.bind('<B1-Motion>', set_rectangle)
        self.__canvas.bind('<ButtonRelease-1>', register_square)

        def delete_last():
            if len(self.__boxes) == 0:
                return
            self.__boxes = self.__boxes[:-1]
            self.__box_labels = self.__box_labels[:-1]
            draw_boxes()

        undo_button = Button(
            self.root, text="Undo", command=delete_last,
            bg=self.__background_color, fg=self.__foreground_color, font=('Arial', 10),
            activebackground=self.__active_background_color, activeforeground=self.__active_foreground_color,
            padx=20
        )
        self.__register_component(undo_button)
        undo_button.grid(row=0, column=0, sticky=EW)

        def next_identifier():
            scaled_boxes = []
            if len(self.__boxes) > 0:
                for box in self.__boxes:
                    scaled_boxes.append((box[0] * self.__width_ratio, box[1] * self.__height_ratio,
                                         box[2] * self.__width_ratio, box[3] * self.__height_ratio))
                self.__boxes_dictionary[identifier] = (scaled_boxes, tuple(self.__box_labels))
                with open(self.__boxes_dictionary_path, 'wb') as f:
                    pickle.dump(self.__boxes_dictionary, f)
            if identifier not in self.__exclude:
                self.__exclude.append(identifier)
            self.__current_index += 1
            if self.__current_index < len(self.__exclude) - 1:
                self.__open_annotation_window(self.__exclude[self.__current_index],
                                              show_already_annotated=show_already_annotated)
            else:
                self.__open_annotation_window(show_already_annotated=show_already_annotated)
            print(len(self.__boxes_dictionary.keys()))

        next_button = Button(
            self.root, text="Next", command=next_identifier,
            bg=self.__background_color, fg=self.__foreground_color, font=('Arial', 10),
            activebackground=self.__active_background_color, activeforeground=self.__active_foreground_color,
            padx=20
        )
        self.__register_component(next_button)
        next_button.grid(row=1, column=0, sticky=EW)

        def previous_identifier():
            scaled_boxes = []
            if len(self.__boxes) > 0:
                for box in self.__boxes:
                    scaled_boxes.append((box[0] * self.__width_ratio, box[1] * self.__height_ratio,
                                         box[2] * self.__width_ratio, box[3] * self.__height_ratio))
                self.__boxes_dictionary[identifier] = (scaled_boxes, tuple(self.__box_labels))
                with open(self.__boxes_dictionary_path, 'wb') as f:
                    pickle.dump(self.__boxes_dictionary, f)
            self.__open_annotation_window(self.__exclude[self.__current_index],
                                          show_already_annotated=show_already_annotated)
            self.__current_index -= 1
            print(len(self.__boxes_dictionary.keys()))

        previous_button = Button(
            self.root, text="Previous", command=previous_identifier,
            bg=self.__background_color, fg=self.__foreground_color, font=('Arial', 10),
            activebackground=self.__active_background_color, activeforeground=self.__active_foreground_color,
            padx=20
        )
        self.__register_component(previous_button)
        previous_button.grid(row=2, column=0, sticky=EW)
        draw_boxes()

    def __open_ground_truth_annotation_window(self, for_trees, identifier=None):
        self.root.title(f'Does this artwork contain {"trees" if for_trees else "persons"}?')
        self.__reset_gui()
        da: DataAccess = DataAccess.instance
        ids = list(da.get_ids())
        random.shuffle(ids)
        if for_trees:
            current_dictionary = self.__contains_trees
            dictionary_path = self.__trees_dictionary_path
        else:
            current_dictionary = self.__contains_persons
            dictionary_path = self.__persons_dictionary_path

        if identifier is None:
            for i in ids:
                if i not in current_dictionary.keys() and i not in self.__exclude:
                    identifier = i
                    break
        if identifier is None:
            self.__open_start_menu()

        image = da.get_PIL_image_from_identifier(identifier=identifier, enhanced=True)
        image = image.resize((800, 800), PIL.Image.LANCZOS)
        image = ImageTk.PhotoImage(image)

        label = Label(self.root, bg=self.__background_color, image=image)
        label.photo = image
        label.grid(row=3, column=0)
        self.__register_component(label)

        def contains_tree(val):
            current_dictionary[identifier] = val
            if len(current_dictionary.keys()) % 100 == 0:
                with open(dictionary_path, 'wb+') as f:
                    pickle.dump(current_dictionary, f)
                print(f'Saved, length = {len(current_dictionary.keys())}')
            next_identifier()

        label.focus()
        label.bind('<Right>', lambda _: contains_tree(True))
        label.bind('<Left>', lambda _: contains_tree(False))

        def delete_last():
            del current_dictionary[list(current_dictionary.keys())[len(current_dictionary.keys()) - 1]]
            with open(dictionary_path, 'wb+') as f:
                pickle.dump(current_dictionary, f)
            previous_identifier()

        undo_button = Button(
            self.root, text="Undo", command=delete_last,
            bg=self.__background_color, fg=self.__foreground_color, font=('Arial', 10),
            activebackground=self.__active_background_color, activeforeground=self.__active_foreground_color,
            padx=20
        )

        self.__register_component(undo_button)
        undo_button.grid(row=0, column=0, sticky=EW)

        def next_identifier():
            if identifier not in self.__exclude:
                self.__exclude.append(identifier)
            self.__current_index += 1
            if self.__current_index < len(self.__exclude) - 1:
                self.__open_ground_truth_annotation_window(identifier=self.__exclude[self.__current_index],
                                                           for_trees=for_trees)
            else:
                self.__open_ground_truth_annotation_window(for_trees=for_trees)
            print(len(current_dictionary.keys()))

        next_button = Button(
            self.root, text="Next", command=next_identifier,
            bg=self.__background_color, fg=self.__foreground_color, font=('Arial', 10),
            activebackground=self.__active_background_color, activeforeground=self.__active_foreground_color,
            padx=20
        )
        self.__register_component(next_button)
        next_button.grid(row=1, column=0, sticky=EW)

        def previous_identifier():
            self.__open_ground_truth_annotation_window(identifier=self.__exclude[self.__current_index],
                                                       for_trees=for_trees)
            self.__current_index -= 1
            print(len(current_dictionary.keys()))

        previous_button = Button(
            self.root, text="Previous", command=previous_identifier,
            bg=self.__background_color, fg=self.__foreground_color, font=('Arial', 10),
            activebackground=self.__active_background_color, activeforeground=self.__active_foreground_color,
            padx=20
        )

        self.__register_component(previous_button)
        previous_button.grid(row=2, column=0, sticky=EW)


if __name__ == '__main__':
    _ = DataAccess.instance
    fc: FilterCache = FilterCache.instance
    _ = fc.get_filtered_identifiers(OriginContainer(('Title', 'Exp')))
    gui_handler: GUIHandler = GUIHandler.instance
    gui_handler.start()
