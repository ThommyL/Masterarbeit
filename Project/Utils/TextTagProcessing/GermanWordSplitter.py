"""
GermanWordSplitter.py
"""

from typing import List, Tuple, Dict, Set, Optional

import numpy as np

from Project.Utils.Misc.Nlp import NLP


class GermanWordSplitter:
    """
    Some longer words are not represented by the language model that is used in this project.
    However, it is reasonable to belief that the words that together form said longer word sufficiently
    represent it. E.g. "Abend" and "Landschaft" together convey the meaning of "Abendlandschaft"
    Note: This method is only used in the data cleaning process and thus the focus was not on efficiency.
    """

    def __init__(self):
        self.__acceptable_to_miss: List[str] = ['n', 'in', 'ist', 'heit', 'iner', 'ende']
        self.__endings: List[str] = ['', 'e', 'es', 'en', 'em', 'er']
        before_endings = ['isch', 'lich', 'artig', 'haft', 'ig']
        tmp = []
        for e in before_endings:
            tmp.append(e + 'st')
        before_endings += tmp
        for elem in before_endings:
            for e in self.__endings:
                self.__acceptable_to_miss.append(elem + e)
        self.__endings += ['d', 's']
        self.__acceptable_to_miss += self.__endings
        self.__acceptable_to_miss.sort(key=len)
        self.__endings.sort(key=len)

        # Note: list of words is filtered version of result of [w for w in nlp.vocab.strings if len(w) == 3
        # and nlp.vocab[w].rank < 100000 and nlp.vocab[w].is_alpha and nlp.vocab[w].shape_ == 'Xxx']
        # 'alt' and 'ein', 'roh', 'Roh', 'Öl', 'Eis', 'Sau', 'Hof', 'Tor', 'Eid', 'Rad', 'Bot', 'Bau', 'Ton', 'Erz',
        # 'Kur', 'Tag', 'Au' and 'Guß' (old form of the word 'Guss') were added also
        self.__valid_short_words: Set[str] = {
            'Kuh', 'See', 'Akt', 'Bär', 'Opa', 'Eva', 'Mut', 'Oma', 'Ehe', 'Eis', 'Maß', 'DDR', 'Rot', 'Box', 'roh',
            'Aas', 'Axt', 'Klo', 'Abt', 'Zoo', 'Dom', 'Süd', 'Ost', 'Hut', 'alt', 'Not', 'Arm', 'Ruf', 'Oma', 'Ehe',
            'Tee', 'Bar', 'Rot', 'Hof', 'Rat', 'Amt', 'Zug', 'Tat', 'Typ', 'Fuß', 'Bau', 'See', 'Ort', 'Weg', 'Sex',
            'Uhr', 'ein', 'roh', 'Roh', 'Öl', 'Eis', 'Sau', 'Hof', 'Tor', 'Eid', 'Rad', 'Bot', 'Bau', 'Ton', 'Erz',
            'Kur', 'Tag', 'Au', 'Guß'}

        # Looking over some of the results, these were found to be filtered out
        # Note: Most of these were collected when the algorithm was not functioning properly, but leaving those in does
        # not hurt

        self.__filter_out: Set[str] = {
            'amal', 'kite', 'chen', 'quiet', 'fire', 'thum', 'poti', 'lund', 'chens', 'schl', 'ende', 'cher', 'erst',
            'stoch', 'keys', 'goss', 'sass', 'eigen', 'goth', 'abel', 'there', 'onen', 'nger', 'verw', 'hard', 'sent',
            'rich', 'lantern', 'child', 'hab', 'ille', 'bole', 'elli', 'seid', 'sien', 'scha', 'sche', 'gale', 'keys',
            'ndung', 'peri', 'dragon', 'sine', 'all', 'versch', 'schen', 'child', 'risch', 'cher', 'bourough', 'past',
            'lough', 'arts', 'iner', 'ffel', 'inde', 'geln', 'ismus', 'sans', 'icht', 'line', 'all', 'mand', 'ling',
            'habe', 'meri', 'ali', 'tern', 'heist', 'corn', 'ward', 'bare', 'anis', 'nder', 'chat', 'pohl', 'elle',
            'coli', 'leigh', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'singh',
            'ales', 'ossi', 'alen', 'ender', 'cken', 'shut', 'ngere', 'thum', 'salle', 'schw', 'bath', 'aten', 'eien',
            'ache', 'vors', 'hacking', 'nimo', 'till', 'sten', 'modi', 'fort', 'perm', 'oser', 'erbil', 'thea', 'trum',
            'schler', 'hove', 'fries', 'keit', 'char', 'iten', 'mein', 'xena', 'tran', 'baia', 'bain', 'meri', 'neri',
            'hora', 'water', 'goss', 'sass', 'port', 'supra', 'mente', 'tter', 'pros', 'sein', 'mein', 'radi', 'tion',
            'dosi', 'quiet', 'inge', 'nicht', 'paint', 'chart', 'ratio', 'ones', 'eien', 'tlich', 'burger', 'bill',
            'iner', 'nahm', 'keys', 'erin', 'hume', 'reith'}

        self.__filter_from_final_result: Set[str] = {
            'ober', 'unter', 'eins', 'zwei', 'drei', 'vier', 'fünf', 'sechs', 'sieben', 'acht', 'neun', 'zehn',
            'vorder', 'hinter', 'hoch', 'nieder', 'erst', 'zweit', 'dritt', 'viert', 'fünft', 'sechst', 'siebt',
            'acht', 'neunt', 'zehnt', 'ein', 'mehr', 'dahin', 'dorthin', 'heran', 'herbei', 'durch', 'reich', 'neben',
            'nahme', 'sein', 'endes', 'voll'}

        # The higher this threshold is, the more filtering will be necessary
        self.__rank_threshold: int = 150000
        self.__nlp = NLP.instance.nlp

    @staticmethod
    def __split_swapped_list_to_word_sizes(
            word_list: List[str], swapped_list: List[bool]) -> List[List[bool]]:
        """
        :param word_list: List of words that together have the length of swapped_list
        :param swapped_list: List of boolean values indicating whether a letter was swapped or not
        :return: Two dimensional list, mapping the values of "swapped_list" to the words of "word_list"
        """
        total_length = 0
        for w in word_list:
            total_length += len(w)
        swapped_split_result: List[List[bool]] = []
        tmp: List[bool] = []
        index: int = 0
        for w in word_list:
            for _ in w:
                tmp.append(swapped_list[index])
                index += 1
            swapped_split_result.append(tmp)
            tmp = []
        return swapped_split_result

    def __find_best_candidate(
            self, candidates: List[List[str]], corrections: List[int]) -> List[str]:
        """
        :param candidates: Different possible results to choose from
        :param corrections: A list containing how many appended letters each option has
        :return: The option that covers the most letters of the original word. In case of equal coverage, the options
        with lower coverage are discarded and the option with the lowest average rank is chosen.
        """
        coverage_dict: Dict[int, int] = dict()
        for index, c in enumerate(candidates):
            coverage = 0
            for element in c:
                coverage += len(element)
            coverage_dict[index] = coverage - corrections[index]

        # Take the option that covers the most letters
        max_val: int = max(coverage_dict.values())
        left_over_candidates: List[List[str]] = []
        for k in coverage_dict.keys():
            if coverage_dict[k] == max_val:
                left_over_candidates.append(candidates[k])

        # In case that multiple options cover the same amount of letters, take the one with the best average rank
        current_best: List[str] = left_over_candidates[0]
        current_ranks: List[int] = [self.__nlp.vocab[element].rank for element in left_over_candidates[0]]
        if len(current_ranks) == 0:
            best_average_rank: float = np.inf
        else:
            best_average_rank: float = sum(current_ranks) / len(current_ranks)
        for c in left_over_candidates[1:]:
            current_ranks: List[int] = [self.__nlp.vocab[element].rank for element in c if
                                        element not in self.__filter_from_final_result]
            if len(current_ranks) == 0:
                current_average_rank = np.inf
            else:
                current_average_rank: float = sum(current_ranks) / len(current_ranks)
            if current_average_rank < best_average_rank:
                best_average_rank: float = current_average_rank
                current_best: List[str] = c
        return current_best

    @staticmethod
    def __bring_found_and_missed_into_order(found_words: List[str], missed_pieces: List[str],
                                            appended_letter: List[bool]) -> Tuple[List[str], List[bool]]:
        """
        :param found_words: List of found words
        :param missed_pieces: List of missed words
        :param appended_letter: List of boolean indicating whether a letter was appended to the words in word_list
        :return: List of found and missed words, such that they are in the right order, as well as a version of
        "appended_letter" that is corrected in a way that there the boolean value "False" is inserted in the places
        corresponding to the entries of "missed_pieces".
        """
        assert len(missed_pieces) == len(found_words) + 1, 'Parameter \'missed_pieces\' needs to have one more ' \
                                                           'element than parameter \'found_words\''
        parts: List[str] = []
        appended_letter_extended: List[bool] = []
        for index in range(len(missed_pieces)):
            if missed_pieces[index] != '':
                appended_letter_extended.append(False)
                parts.append(missed_pieces[index])
            if index < len(found_words):
                parts.append(found_words[index])
                appended_letter_extended.append(appended_letter[index])
        return parts, appended_letter_extended

    def __change_ending_and_case(self, word: str, append_letter: bool) -> Optional[str]:
        """
        :param word: Word which should be possibly corrected
        :param append_letter: Whether the letter 'e' should possibly be appended to the word or not.
        :return: The best ranking option of the word, whereas the options are the word with possible endings removed, as
        well as a version of the word with the letter 'e' appended, if the parameter "append_letter" is set to true.
        """
        options = [word]

        temp = []

        # e.g. "Hochschule" instead of "Hochschul"
        if append_letter and self.__nlp.vocab[word + 'e'].rank < self.__nlp.vocab[word].rank:
            temp.append(word + 'e')

        for ending in self.__endings:
            if ending == '':
                continue
            if word.endswith(ending):
                temp.append(word[:-len(ending)])
        for o in options:
            temp.append(o.lower())
        for o in options:
            temp.append(o.capitalize())
        options += temp
        options = list(set(options))

        options = [o for o in options if (len(o) > 3 or o in self.__valid_short_words)
                   and o.lower() not in self.__filter_out and sum(self.__nlp(o).vector) != 0]

        if len(options) == 0:
            return None
        return self.__find_best_candidate([[o] for o in options], [0 for _ in options])[0]

    def __fix_and_validate_words(self, word_to_check: str, found: List[str], swapped: List[bool],
                                 append_letter: bool, appended: List[bool]
                                 ) -> Tuple[List[str], List[str], List[bool], List[bool]]:
        """
        :param word_to_check: Original word to compare against
        :param found: Words that were found in previous steps
        :param swapped: A list of boolean values, indicating whether a letter was swapped or not
        :param append_letter: A boolean value indicating whether words should possibly be appended or not
        :param appended: A list of boolean values, indicating whether a word in "found" was appended or not
        :return: A List of found words; a list of missing pieces; a list of boolean values indicating whether a letter
        was swapped or not; A list of boolean values, indicating whether a word was appended or not
        """
        # Check which letters were left out
        all_missed: List[str] = []
        current_word: str = word_to_check

        # If this letter was swapped out, it means that a valid word was found covering this letter
        # Therefore it is save to swap it, since it will not end up in the "all_missed" list
        for j in range(len(swapped) - 1):
            if swapped[j]:
                current_word = current_word[:j] + 'e' + current_word[j + 1:]

        if len(swapped) > 0 and swapped[len(swapped) - 1]:
            current_word = current_word[:-1] + 'e'

        for r_index, r in enumerate(found):
            missed: str = ''
            corrected = r
            if appended[r_index]:
                corrected = corrected[:-1]
            while len(current_word) > 0:
                if current_word.lower().startswith(corrected.lower()):
                    current_word = current_word[len(corrected):]
                    break

                missed += current_word[0]
                current_word = current_word[1:]
            all_missed.append(missed)
        all_missed.append(current_word)

        if all_missed[0].lower() not in self.__acceptable_to_miss:
            return found, all_missed, swapped, appended

        # Note: Only trying to append to words
        try_to_merge_again: List[str] = all_missed[1:]

        updated: bool = True
        processed = [False for _ in try_to_merge_again]
        while updated:
            updated = False
            for j, merge_with in enumerate(try_to_merge_again):
                if processed[j] or merge_with == '':
                    continue
                processed[j] = True
                base_word_candidate = found[j] + merge_with
                word_candidate: Optional[str] = self.__change_ending_and_case(
                    word=base_word_candidate, append_letter=append_letter)

                current_rank: int = np.inf

                # Note: The rank of the word that would be computed from the base word later
                if word_candidate is not None and sum(self.__nlp(word_candidate).vector) != 0 and \
                        self.__nlp.vocab[word_candidate].rank < self.__rank_threshold:
                    current_rank = self.__nlp.vocab[word_candidate].rank

                if current_rank == np.inf:
                    continue

                found[j] = word_candidate
                appended[j] = appended[j] or word_candidate == base_word_candidate + 'e'
                updated = True
                try_to_merge_again[j] = base_word_candidate[len(word_candidate):]
                break

        missing = [all_missed[0]] + try_to_merge_again
        not_appended: List[str] = []

        for j in range(len(appended)):
            if appended[j]:
                not_appended.append(found[j][:-1])
            else:
                not_appended.append(found[j])

        ordered_parts, _ = self.__bring_found_and_missed_into_order(not_appended, missing, appended)
        tmp_word = ''
        for result_word in ordered_parts:
            tmp_word += result_word

        new_was_swapped: List[bool] = []
        for letter_1, letter_2 in zip(word_to_check.lower(), tmp_word.lower()):
            new_was_swapped.append(letter_1 != letter_2)

        index = 0
        for j in range(len(appended)):
            index += len(found[j]) - 1
            if appended[j]:
                new_was_swapped.insert(index + 2, False)
        return found, missing, new_was_swapped, appended

    def __validate_words(self, actual_word: str, changed_word: str):
        """
        :param actual_word: An unchanged word
        :param changed_word: The result of the method "change_ending_and_case" with the input of the parameter
        "actual_word". Note: This could be recalculated in this method, but since the calling methods have this value
        available, a parameter is used here in order to improve efficiency.
        :return: True if these words are valid, False otherwise
        """
        if changed_word is not None and \
                (len(actual_word) > 3 or actual_word in self.__valid_short_words) and \
                actual_word.lower() not in self.__filter_out and changed_word.lower() not in self.__filter_out and \
                sum(self.__nlp(changed_word).vector) != 0 and self.__nlp.vocab[changed_word].rank < self.__rank_threshold:
            return True
        return False

    def __recombine_words(self, word_list: List[str], swapped: List[bool], append_letter: bool,
                          appended: List[bool]
                          ) -> Tuple[List[str], List[str], List[bool], List[bool]]:
        """
        :param word_list: Words that were found in previous steps
        :param swapped: A list of boolean values, indicating whether a letter was swapped or not
        :param append_letter: A boolean value indicating whether words should possibly be appended or not
        :param appended: A list of boolean values, indicating whether a word in "found" was appended or not
        :return: A List of found words; a list of missing pieces; a list of boolean values indicating whether a letter
        was swapped or not; A list of boolean values, indicating whether a word was appended or not
        """
        updated: bool = True
        word_list_corrected: List[str] = []
        word_sized_swapped: List[List[bool]] = self.__split_swapped_list_to_word_sizes(word_list, swapped)
        valid_words: List[bool] = []

        for word in word_list:
            new_word: Optional[str] = self.__change_ending_and_case(word, append_letter=append_letter)
            if new_word is not None and len(new_word) > len(word):
                new_word = word
            if self.__validate_words(word, new_word):
                valid_words.append(True)
            else:
                valid_words.append(False)

        current_index_for_swapped: int = 0
        for word_index, word in enumerate(word_list):
            current_word = ''
            for letter_index, current_letter in enumerate(word):
                if letter_index == len(word) - 1 and appended[word_index]:
                    continue
                if swapped[current_index_for_swapped]:
                    current_word += 's'
                else:
                    current_word += current_letter
                current_index_for_swapped += 1
            word_list_corrected.append(current_word)

        while updated:
            updated = False
            word_list_corrected_copy: List[str] = [element for element in word_list_corrected]
            appended_copy: List[bool] = [element for element in appended]
            word_sized_swapped_copy: List[List[bool]] = [element for element in word_sized_swapped]
            valid_words_copy: List[bool] = [element for element in valid_words]

            results: List[
                Optional[List[
                    List[str], List[bool], List[List[bool]], List[bool]]
                ]
            ] = [None, None, None]

            for j in range(1, len(word_list_corrected)):
                for merge_index, to_merge in enumerate(((j - 1, j), (j, j + 1))):
                    if to_merge[1] == len(word_list_corrected):
                        continue

                    for a in self.__acceptable_to_miss:
                        try_to_add: str = word_list_corrected[to_merge[1]].lower()

                        if not try_to_add.endswith(a):
                            continue

                        cut_at: int = -len(a)
                        # treating special case: -0 equals 0, therefore the special syntax with the minus sign
                        # does not work
                        if cut_at == 0:
                            cut_at = len(try_to_add)
                        try_to_add = try_to_add[:cut_at]

                        new_word_actual = word_list_corrected[to_merge[0]] + try_to_add.lower()
                        new_word: Optional[str] = self.__change_ending_and_case(
                            new_word_actual, append_letter=append_letter)
                        if new_word is not None and len(new_word) > len(new_word_actual):
                            new_word = new_word_actual

                        if self.__validate_words(new_word_actual, new_word):
                            word_list_corrected[to_merge[0]] = new_word_actual
                            appended[to_merge[0]] = False
                            word_sized_swapped[to_merge[0]] += word_sized_swapped[to_merge[1]]
                            word_sized_swapped[to_merge[0]] = word_sized_swapped[to_merge[0]][:len(new_word_actual)]
                            valid_words[to_merge[0]] = True

                            del word_list_corrected[to_merge[1]]
                            del appended[to_merge[1]]
                            del word_sized_swapped[to_merge[1]]
                            del valid_words[to_merge[1]]

                            updated = True
                            results[merge_index] = [word_list_corrected, appended, word_sized_swapped, valid_words]
                            word_list_corrected = [element for element in word_list_corrected_copy]
                            appended = [element for element in appended_copy]
                            word_sized_swapped = [element for element in word_sized_swapped_copy]
                            valid_words = [element for element in valid_words_copy]
                            break

                if results[0] is None and results[1] is None and j + 1 != len(word_list_corrected):
                    word_beginning = word_list_corrected[j - 1]
                    if appended[j - 1]:
                        word_beginning = word_beginning[:-1]
                    word_middle = word_list_corrected[j].lower()
                    if appended[j]:
                        word_middle = word_middle[:-1]
                    word_ending = word_list_corrected[j + 1].lower()
                    if appended[j + 1]:
                        word_ending = word_ending[:-1]

                    new_word_actual = word_beginning + word_middle + word_ending
                    new_word = self.__change_ending_and_case(word=new_word_actual, append_letter=append_letter)
                    if new_word is not None and len(new_word_actual) > len(new_word):
                        new_word = new_word_actual
                    if self.__validate_words(new_word_actual, new_word):
                        word_list_corrected[j - 1] = new_word_actual
                        appended[j - 1] = False
                        word_sized_swapped[j - 1] = word_sized_swapped[j - 1] + word_sized_swapped[j] + \
                            word_sized_swapped[j + 1]
                        word_sized_swapped[j - 1] = word_sized_swapped[j - 1][:len(new_word_actual)]
                        valid_words[j - 1] = True

                        # Deleting indices j and j + 1
                        del word_list_corrected[j]
                        del word_list_corrected[j]
                        del appended[j]
                        del appended[j]
                        del word_sized_swapped[j]
                        del word_sized_swapped[j]
                        del valid_words[j]
                        del valid_words[j]

                        updated = True
                        results[2] = [word_list_corrected, appended, word_sized_swapped, valid_words]
                        word_list_corrected = [element for element in word_list_corrected_copy]
                        appended = [element for element in appended_copy]
                        word_sized_swapped = [element for element in word_sized_swapped_copy]
                        valid_words = [element for element in valid_words_copy]

                merge_only_with_first: bool = False
                merge_only_with_second: bool = False
                merge_with_both: bool = results[2] is not None

                for index in range(len(results)):
                    if results[index] is None:
                        continue

                    results[index] = [r for r in results[index] if r is not None]

                if results[1] is None and results[0] is not None:
                    merge_only_with_first = True
                elif results[0] is None and results[1] is not None:
                    merge_only_with_second = True
                elif results[0] is not None and results[1] is not None:
                    if sum([self.__nlp.vocab[self.__change_ending_and_case(
                            word=element, append_letter=append_letter) or element].rank for element in
                            results[0][0]]) < sum([self.__nlp.vocab[element].rank for element in results[1][0]]):
                        merge_only_with_first = True
                    else:
                        merge_only_with_second = True

                merge_index: int = -1
                if merge_only_with_first:
                    merge_index = 0
                elif merge_only_with_second:
                    merge_index = 1
                elif merge_with_both:
                    merge_index = 2

                if merge_index != -1:
                    word_list_corrected = results[merge_index][0]
                    # noinspection PyTypeChecker
                    appended: List[bool] = results[merge_index][1]
                    # noinspection PyTypeChecker
                    word_sized_swapped: List[List[bool]] = results[merge_index][2]
                    # noinspection PyTypeChecker
                    valid_words: List[bool] = results[merge_index][3]
                    updated = True

                if updated:
                    break

        if len(word_list_corrected) == 0:
            return [], word_list, swapped, appended
        assert word_list_corrected[0] is not None, 'Internal Error'

        word_list_corrected_tmp: List[str] = []
        for word_index, word in enumerate(word_list_corrected):
            current_word = ''
            for letter_index, current_letter in enumerate(word):
                if word_sized_swapped[word_index][letter_index]:
                    if letter_index != len(word):
                        current_word += 'e'
                    else:
                        current_word += current_letter
                else:
                    current_word += current_letter
            word_list_corrected_tmp.append(current_word)
        word_list_corrected = word_list_corrected_tmp

        word_list_appended = [w for w in word_list_corrected]
        for j in range(len(appended)):
            if appended[j]:
                word_list_appended[j] += 'e'

        # Reducing to a one dimensional list
        swapped = []
        for s in word_sized_swapped:
            for val in s:
                swapped.append(val)

        still_missing: List[str] = []
        for j in range(len(valid_words)):
            if not valid_words[j]:
                still_missing.append(word_list_corrected[j])
        return word_list_corrected, still_missing, swapped, appended

    def __split_word(self, word_to_split: str, capitalize: bool, swap_letters: bool, swapped: List[bool],
                     append_letter: bool, appended: List[bool]
                     ) -> Tuple[List[str], List[bool], List[bool]]:
        """
        :param word_to_split: Word that is to be split up into pieces
        :param capitalize: Whether the first letter of the word should be capitalized or not
        :param swap_letters: Whether letters should possibly be swapped or not
        :param swapped: array of boolean values of length len(word_to_split), indicating whether letters have been
        swapped already or not
        :param append_letter: Whether letters should possibly be appended to a word or not
        :param appended: a list of boolean values indicating whether words in "word_to_split" have been appended or not
        :return: A list of words that are contained in "word_to_split", which the language model recognized; a list of
        boolean values indicating whether a letter has been swapped or not; a list of boolean values, indicating wheter
        a word has been appended or not.
        """
        assert len(word_to_split) == len(swapped), 'The parameter \'word_to_split\' must be of the same length as ' \
                                                   'the parameter \'swapped\'.'

        if len(word_to_split) <= 2:
            return [], swapped, appended

        if capitalize and word_to_split[0] != 'ß':
            word_to_split = word_to_split.capitalize()

        found_word: Optional[str] = None
        found_word_without_additional_ending: Optional[str] = None
        new_word: str = ''
        skip_maybe: int = 0
        current_swapped_list: List[bool] = [s for s in swapped]
        extra_letter = None
        for index, l in enumerate(word_to_split):
            new_word += l
            options: List[str] = [new_word]
            option_ranks = [None, None]
            options_skip_maybe = [skip_maybe, skip_maybe]
            options_swapped = [[s for s in swapped]]
            options_extra_letter = [False]

            # e.g. Gebirge instead of Gebirgs
            if swap_letters and new_word.endswith('s'):
                options.append(new_word[:-1] + 'e')
                tmp = [s for s in swapped]
                tmp[index] = True
                options_swapped.append(tmp)
                options_extra_letter.append(False)
            for option_index, o in enumerate(options):
                corrected_option = self.__change_ending_and_case(word=o, append_letter=append_letter) or o
                if len(o) > len(corrected_option):
                    corrected_option = o

                if o.lower() not in self.__filter_out and (len(o) > 3 and sum(self.__nlp(corrected_option).vector) != 0) \
                        or (o in self.__valid_short_words):
                    if self.__nlp.tokenizer(o)[0].lemma_ == found_word:
                        options_skip_maybe[option_index] += 1
                    else:
                        if self.__nlp.vocab[corrected_option].rank < self.__rank_threshold:
                            options_skip_maybe[option_index] = 0
                            if o.lower() not in self.__acceptable_to_miss and o.lower() not in self.__filter_out:

                                # Allow words that are the same as the previously found word, but with a longer ending
                                found_ending: Optional[str] = None
                                if found_word is not None:
                                    for ending in self.__endings:
                                        if found_word_without_additional_ending is None:
                                            found_word_without_additional_ending = found_word
                                        if found_word_without_additional_ending + ending == corrected_option:
                                            found_ending = ending
                                            break

                                # It is preferable to have "Abend", "Land", "Schaft" than "Abendland", "Schaft", since
                                # word and missing pieces are recombined later anyway
                                if found_word is None or found_ending is not None or \
                                        self.__nlp.vocab[corrected_option].rank < self.__nlp.vocab[found_word].rank:
                                    option_ranks[option_index] = self.__nlp.vocab[corrected_option].rank
                                    options_extra_letter[option_index] = corrected_option == o + 'e'
                                    if found_ending is not None:
                                        found_word_without_additional_ending = found_word[:-len(found_ending)]
                                        options_skip_maybe[option_index] = len(found_ending)

            if option_ranks[0] is None and option_ranks[1] is None:
                continue

            correct_index: int = 1
            if option_ranks[1] is None:
                correct_index = 0
            elif option_ranks[0] is None:
                correct_index = 1
            else:
                if option_ranks[0] > option_ranks[1]:
                    correct_index = 1

            found_word = options[correct_index]
            skip_maybe = options_skip_maybe[correct_index]
            current_swapped_list = options_swapped[correct_index]
            extra_letter = options_extra_letter[correct_index]
        swapped = current_swapped_list

        if found_word is None:
            recursive_result: List[str]
            recursive_swapped_list: List[bool]
            recursive_result, recursive_swapped_list, appended = self.__split_word(
                word_to_split=word_to_split[1:], capitalize=capitalize, swap_letters=swap_letters,
                swapped=[False for _ in word_to_split[1:]], append_letter=append_letter, appended=appended)

            for j in range(1, len(swapped)):
                swapped[j] = recursive_swapped_list[j - 1]

            return recursive_result, swapped, appended

        assert extra_letter is not None, 'Internal Error'

        if len(found_word) != len(word_to_split):
            candidates: List[List[str]] = []
            candidates_swapped: List[List[bool]] = []
            candidates_appended: List[List[bool]] = []
            candidates_skipped: List[int] = []

            for a in self.__acceptable_to_miss:
                if word_to_split[len(found_word):].startswith(a):
                    skip_maybe = max(skip_maybe, len(a))

            for j in range(skip_maybe + 1):
                word_to_try: str = word_to_split[len(found_word):][j:]
                cur_result: Tuple[List[str], List[bool], List[bool]] = self.__split_word(
                    word_to_split=word_to_try, capitalize=capitalize, swap_letters=swap_letters,
                    swapped=[False for _ in word_to_try], append_letter=append_letter, appended=appended)
                if len(cur_result[0]) == 0:
                    continue

                candidates.append(cur_result[0])
                candidates_swapped.append(cur_result[1])
                candidates_appended.append(cur_result[2])
                candidates_skipped.append(j + 1)

            if len(candidates) > 0:
                appended_corrections: List[int] = [0 for _ in candidates]
                for j in range(len(appended_corrections)):
                    current_correction = 0
                    for ap in candidates_appended[j]:
                        if ap:
                            current_correction += 1
                    appended_corrections[j] = current_correction

                best_candidate: List[str] = self.__find_best_candidate(candidates=candidates,
                                                                       corrections=appended_corrections)
                candidate_index: int = -1
                for j in range(len(candidates)):
                    if candidates[j] == best_candidate:
                        candidate_index = j
                        break
                assert candidate_index != -1, 'Internal Error'
                new_swapped_list: List[bool] = candidates_swapped[candidate_index][
                                                    candidates_skipped[candidate_index]:]
                swapped = swapped[:len(swapped) - len(new_swapped_list)] + new_swapped_list
                return [found_word] + best_candidate, swapped, [extra_letter] + candidates_appended[candidate_index]
            else:
                return [found_word], swapped, appended + [extra_letter]
        return [found_word], swapped, appended + [extra_letter]

    def split_german_word(self, word: str) -> List[str]:
        """
        :param word: German word to split
        :return: A List of words that are represented in the model that is used and are (likely) related to the words
        that make up the input word.
        """
        if len(word) == 0:
            return []
        words: List[str] = []
        current_word: str = ''
        for letter in word:
            if current_word == '':
                current_word = letter
            elif letter.isupper():
                words.append(current_word)
                current_word = letter
            else:
                current_word += letter
        words.append(current_word)

        if len(words) != 1:
            found_words: List[str] = []
            for w in words:
                found_words += self.split_german_word(word=w)
            return found_words

        capitalize: bool = word[0].isupper()
        candidates: List[List[str]] = []
        number_appended: List[int] = []

        swap_options: List[bool] = [False]
        append_options = [True, False]

        if capitalize:
            swap_options.append(True)

        for to_append in append_options:
            for swap in swap_options:
                current_result: List[str]
                swapped: List[bool]

                current_result, swapped, appended_letter = self.__split_word(
                    word_to_split=word, capitalize=capitalize, swap_letters=swap, swapped=[False for _ in word],
                    append_letter=to_append, appended=[])
                swapped_split = self.__split_swapped_list_to_word_sizes(current_result, swapped)

                swapped_index = 0
                for swapped_split_index, zipped in enumerate(zip(swapped_split, appended_letter)):
                    swapped_part, cur_append = zipped
                    swapped_index += len(swapped_split) - 1
                    if cur_append:
                        # If the last letter was swapped, then do not append an e
                        if swapped_part[len(swapped_part) - 1]:
                            raise Exception(
                                f'Tried to append a letter on a word where the last letter was swapped. Word: {word}')
                        else:
                            current_result[swapped_split_index] += 'e'
                            swapped.insert(swapped_index + len(current_result[swapped_split_index]) + 1, False)

                still_missed: List[str]
                current_result, still_missed, swapped, appended_letter = self.__fix_and_validate_words(
                    word_to_check=word, found=current_result, swapped=swapped, append_letter=to_append,
                    appended=appended_letter)

                ordered_parts, ordered_appended = \
                    self.__bring_found_and_missed_into_order(current_result, still_missed, appended_letter)

                current_result, still_missed, swapped, appended_letter = self.__recombine_words(
                    word_list=ordered_parts, swapped=swapped, append_letter=to_append,
                    appended=ordered_appended)

                for i in range(len(appended_letter)):
                    if appended_letter[i]:
                        current_result[i] += 'e'

                for t in current_result:
                    if len(t) <= 3 and t not in self.__valid_short_words or t.lower() in self.__filter_out:
                        still_missed.append(t)

                valid: bool = True

                for m in still_missed:
                    if m != '' and m not in self.__filter_from_final_result and m not in self.__acceptable_to_miss:
                        valid = False
                        break

                if valid:
                    current_result = [self.__change_ending_and_case(t, to_append) for t in current_result]
                    for t in current_result:
                        if t is None:
                            valid = False
                            break
                if valid:
                    current_result = [t for t in current_result if sum(self.__nlp(t).vector) != 0]
                    candidates.append(
                        [w for w in current_result if w.lower() not in self.__filter_from_final_result and w != ''])
                    nr_appended = 0
                    for ap in appended_letter:
                        if ap:
                            nr_appended += 1
                    number_appended.append(nr_appended)
                else:
                    candidates.append([])
                    number_appended.append(0)

        final_result = self.__find_best_candidate(candidates, number_appended)

        for i in range(len(final_result)):
            # Special cases where the lemmatizer got it wrong
            if not (final_result[i] == 'Paar' or final_result[i] == 'Herde'):
                lemmatized = self.__nlp.tokenizer(final_result[i])[0].lemma_
                if self.__nlp.vocab[lemmatized].rank < self.__nlp.vocab[final_result[i]].rank:
                    final_result[i] = lemmatized
        if len(final_result) == 0 and capitalize:
            return self.split_german_word(word=word.lower())
        return final_result
