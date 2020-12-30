# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np

KP_CONCAT_TYPES = ['one2one', 'random',
                   'pres_abs', 'abs_pres',
                   'nosort', 'nosort_reverse',
                   'alphab', 'alphab_reverse',
                   'length', 'length_reverse']

KP_DATASET_FIELDS = {'scipaper': ('title', 'abstract', 'keywords', None),
                     'qa': ('title', 'question', 'tags', 'categories'),
                     'webpage': ('url', 'text', 'KeyPhrases', None),
                     'news': ('title', 'abstract', 'keyword', 'categories'),
                     'wiki': (None, 'text', None, None)}



def parse_src_fn(ex_dict, title_field, text_field):
    concat_str = ex_dict[title_field] + ' . ' + ex_dict[text_field]
    return concat_str


def kpdict_parse_fn(ex_dict, tokenizer, kp_concat_type, dataset_type='scipaper', max_target_phrases=-1, lowercase=False):
    assert dataset_type in KP_DATASET_FIELDS
    title_field, text_field, keyword_field, category_field = KP_DATASET_FIELDS[dataset_type]

    src_str = parse_src_fn(ex_dict, title_field, text_field)
    if isinstance(ex_dict[keyword_field], str):
        tgt_kps = ex_dict[keyword_field].split(';')
    else:
        tgt_kps = ex_dict[keyword_field]
    if kp_concat_type == 'one2one':
        # sample one tgt from multiple tgts and use it as the only tgt
        rand_idx = np.random.randint(len(tgt_kps))
        tgt_str = tgt_kps[rand_idx]
    elif kp_concat_type in KP_CONCAT_TYPES:
        # generate one2seq training data points
        order = obtain_sorted_indices(src_str.lower().split(),
                                      [kp.lower().split() for kp in tgt_kps],
                                      sort_by=kp_concat_type)
        if max_target_phrases > 0 and len(order) > max_target_phrases:
            order = order[: max_target_phrases]
        tgt = [tgt_kps[idx] for idx in order]
        tgt_str = tokenizer.sep_token.join(tgt)
    else:
        raise NotImplementedError('Unsupported target concatenation type ' + kp_concat_type)

    if lowercase:
        return src_str.lower(), tgt_str.lower()
    return src_str, tgt_str


def obtain_sorted_indices(src, tgt_seqs, sort_by):
    """
    :param src: used for verbatim and alphabetical
    :param tgt_seqs:
    :param sort_by:
    :return:
    """
    num_tgt = len(tgt_seqs)

    if sort_by == 'random':
        sorted_id = np.random.permutation(num_tgt)
    elif sort_by.startswith('nosort'):
        sorted_id = list(range(len(tgt_seqs)))
    elif sort_by.startswith('alphab'):
        sorted_tgts = sorted(enumerate(tgt_seqs), key=lambda x: '_'.join(x[1]))
        sorted_id = [t[0] for t in sorted_tgts]
    elif sort_by.startswith('length'):
        sorted_tgts = sorted(enumerate(tgt_seqs), key=lambda x: len(x[1]))
        sorted_id = [t[0] for t in sorted_tgts]
    elif sort_by == 'pres_abs' or sort_by == 'abs_pres':
        # obtain present flags as well their positions, lowercase should be done beforehand
        present_tgt_flags, present_indices, _ = if_present_duplicate_phrases(src, tgt_seqs)
        # separate present/absent phrases
        present_tgt_idx = np.arange(num_tgt)[present_tgt_flags]
        absent_tgt_idx  = [t_id for t_id, present in zip(range(num_tgt), present_tgt_flags) if ~present]
        absent_tgt_idx  = np.random.permutation(absent_tgt_idx)
        # sort present phrases by their positions
        present_indices = present_indices[present_tgt_flags]
        present_tgt_idx = sorted(zip(present_tgt_idx, present_indices), key=lambda x: x[1])
        present_tgt_idx = [t[0] for t in present_tgt_idx]

        if sort_by == 'pres_abs':
            sorted_id = np.concatenate((present_tgt_idx, absent_tgt_idx), axis=None)
        elif sort_by == 'abs_pres':
            sorted_id = np.concatenate((absent_tgt_idx, present_tgt_idx), axis=None)
        else:
            raise NotImplementedError('Unsupported sort_by value: ' + sort_by)
            sorted_id = present_tgt_idx
    else:
        raise NotImplementedError('Unsupported sort_by value: ' + sort_by)

    if sort_by.endswith('reverse'):
        sorted_id = sorted_id[::-1]

    return np.asarray(sorted_id, dtype=int)


def if_present_duplicate_phrases(src_seq, tgt_seqs):
    """if_present_duplicate_phrases
    Check if each given target sequence verbatim appears in the source sequence
    :param src_seq:
    :param tgt_seqs:
    :param lowercase:
    :param check_duplicate:
    :return:
    """
    present_indices = []
    present_flags = []
    duplicate_flags = []
    phrase_set = set()  # some phrases are duplicate after stemming, like "model" and "models" would be same after stemming, thus we ignore the following ones

    for tgt_seq in tgt_seqs:
        # check if the phrase appears in source text
        # iterate each word in source
        match_flag, match_pos_idx = if_present_phrase(src_seq, tgt_seq)

        # if it reaches the end of source and no match, means it doesn't appear in the source
        present_flags.append(match_flag)
        present_indices.append(match_pos_idx)

        # check if it is duplicate
        if '_'.join(tgt_seq) in phrase_set:
            duplicate_flags.append(True)
        else:
            duplicate_flags.append(False)
        phrase_set.add('_'.join(tgt_seq))

    assert len(present_flags) == len(present_indices)

    return np.asarray(present_flags), \
           np.asarray(present_indices), \
           np.asarray(duplicate_flags)


def if_present_phrase(src_str_tokens, phrase_str_tokens):
    """

    :param src_str_tokens: a list of strings (words) of source text
    :param phrase_str_tokens: a list of strings (words) of a phrase
    :return:
    """
    match_flag = False
    match_pos_idx = -1
    for src_start_idx in range(len(src_str_tokens) - len(phrase_str_tokens) + 1):
        match_flag = True
        # iterate each word in target, if one word does not match, set match=False and break
        for seq_idx, seq_w in enumerate(phrase_str_tokens):
            src_w = src_str_tokens[src_start_idx + seq_idx]
            if src_w != seq_w:
                match_flag = False
                break
        if match_flag:
            match_pos_idx = src_start_idx
            break

    return match_flag, match_pos_idx
