# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import random
import re
import string
import warnings

import numpy as np

import spacy
spacy_nlp = spacy.load('en_core_web_sm')

from fairseq.data import data_utils

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

geometric_p = 0.2
max_phrase_len = 8
span_len_opts = list(range(1, max_phrase_len + 1))
len_distrib = [geometric_p * (1 - geometric_p) ** (i - 1) for i in
               range(1, max_phrase_len + 1)] if geometric_p >= 0 else None
len_distrib = [x / (sum(len_distrib)) for x in len_distrib]

def random_span_parse_fn(ex, sep_token,
                         num_spans=None,
                         max_target_phrases=8,
                         return_masked_source=True,
                         seed=0):
    """
    :param ex:
    :param num_spans: if set, will sample this many spans, otherwise it samples a random number of spans
    :param sep_token:
    :param max_target_phrases:
    :param lowercase:
    :return:
    """
    assert max_target_phrases > 0, 'max_target_phrases must be a positive integer'
    src_text = ex['source']

    with utils.numpy_seed(seed):
        # mask random spans
        src_tokens = src_text.split()

        span_lens = []
        if not num_spans:
            num_spans = np.random.random_integers(max_target_phrases)
        for i in range(num_spans):
            span_len = max(1, np.random.choice(span_len_opts, p=len_distrib))
            span_lens.append(span_len)

        span_lens = sorted(span_lens, reverse=True)  # ensure larger spans get processed first

        spans = []
        is_masked = [False] * len(src_tokens)
        span_idx = 0
        num_try, max_try = 0, len(span_lens) * 4
        while span_idx < len(span_lens):
            # in case there's not much unmasked tokens left
            num_try += 1
            if num_try > max_try: break
            # sample a span start
            span_len = span_lens[span_idx]
            span_left = np.random.random_integers(low=0, high=len(src_tokens)-span_len)
            # some tokens have been masked, skip. Also to ensure no two spans are contiguous
            has_overlap = False
            l_idx = span_left-1 if span_left > 0 else 0
            r_idx = span_left+span_len+1 if span_left+span_len+1 < len(src_tokens) else len(src_tokens)
            for i in range(l_idx, r_idx):
                if is_masked[i]: has_overlap = True
            if has_overlap: continue
            # a new span
            spans.append((span_left, span_left + span_len))
            for i in range(span_len):
                is_masked[span_left+i] = True
            span_idx += 1

        spans = sorted(spans, key=lambda k:k[0]) # order spans by their positions
        masked_src_tokens, infill_spans = [], []
        prev_span_end = 0
        for s in spans:
            masked_src_tokens.extend(src_tokens[prev_span_end: s[0]])
            masked_src_tokens.append('<infill>')
            infill_spans.append(src_tokens[s[0]: s[1]])
            prev_span_end = s[1]
        masked_src_tokens.extend(src_tokens[prev_span_end:])

        span_texts = [' '.join(s) for s in infill_spans]
        if return_masked_source:
            src_text = ' '.join(masked_src_tokens)
        else:
            src_text = src_text

    tgt_text = sep_token.join(span_texts)

    if lowercase:
        return src_text.lower(), tgt_text.lower(), [p.lower() for p in span_texts]

    return src_text, tgt_text, span_texts


def maybe_replace_target(example, label_sample_ratio,
                         max_target_phrases, max_phrase_len=-1,
                         add_control_prefix_prob=0.0,
                         fix_target_number=False, allow_duplicate=False,
                         sep_token='<sep>', seed=0):
    '''
    If additional target label sets are given, we replace example['target'] with new labels
    :param example:
    :param label_sample_ratio: sampling ratio of each extra label set
    :param max_target_phrases:
    :param add_control_prefix_prob: if given, we append the number of phrases as a prefix to source string
    :param fix_target_number: if True, target always contains `max_target_phrases` phrases, otherwise it's sampled in (0, max_target_phrases]
    :param allow_duplicate: if True, target can contain duplicate phrases, otherwise duplicate phrases are removed
    :param seed:
    :return:
    '''
    if not label_sample_ratio: # label set is not given, directly return
        return example
    if max_target_phrases < 0:
        max_target_phrases = 100000 # a very large number
    src_str, tgt_str = example['source'], example['target']
    with data_utils.numpy_seed(seed):
        tgts = []
        for labelset_id, ratio in enumerate(label_sample_ratio):
            candicate_tgts = example['target%d' % labelset_id]
            if isinstance(candicate_tgts, list):
                # ensure each phrase has less than 70 characters and max_phrase_len words
                if max_phrase_len > 0:
                    candicate_tgts = [p for p in candicate_tgts
                                        if len(p) < 70 and len(re.findall(r"\w+|[^\w\s]", p, re.UNICODE)) <= max_phrase_len]
                # remove punctuations
                candicate_tgts = [p.strip() for p in candicate_tgts if len(p.strip()) > 0]

                # determine number of phrases to sample
                if len(candicate_tgts) == 0:
                    continue
                if max_target_phrases < 0:
                    num_to_sample = len(candicate_tgts)
                else:
                    num_to_sample = min(len(candicate_tgts), int(ratio * max_target_phrases))
                if num_to_sample == 0:
                    num_to_sample = 1

                tgts.extend(np.random.choice(candicate_tgts, num_to_sample, replace=False))
            elif isinstance(candicate_tgts, str) and candicate_tgts == '__annotated_kp':
                # ground-truth keyphrases
                assert 'keywords_tokens' in example, 'keywords_tokens not found in example, ' \
                                                     'please ensure the keyphrase transform has run precedingly in the pipeline'
                candicate_tgts = example['keywords_tokens']
                candicate_tgts = [' '.join(p) for p in candicate_tgts]
                if len(candicate_tgts) == 0:
                    continue
                num_to_sample = max(1, min(len(candicate_tgts), int(ratio * max_target_phrases)))
                candicate_tgts = np.random.choice(candicate_tgts, num_to_sample, replace=False)
                np.random.shuffle(candicate_tgts)
            elif isinstance(candicate_tgts, str) and candicate_tgts == '__random_span':
                num_to_sample = max(1, int(ratio * max_target_phrases))
                if num_to_sample > 20:
                    warnings.warn(
                        "current number of random span is %d, please ensure that max_target_phrases is properly set, rather than -1",
                        RuntimeWarning)
                # random spans
                src_str, tgt_str, candicate_tgts = random_span_parse_fn(example, sep_token=sep_token, num_spans=num_to_sample, seed=seed)
            else:
                raise NotImplementedError('Not supported type:' + candicate_tgts)

            tgts.extend(candicate_tgts)

    # deduplicate
    if not allow_duplicate:
        tgts = list(set(tgts))

    # invalid example, will be discarded later
    if len(tgts) == 0:
        example['target'] = ''
        return example

    # print(len(tgts))
    # print(tgts)
    # shuffle order and randomize target size, disabled since random span positions should align with input
    # np.random.shuffle(tgts)

    if not fix_target_number:
        tgts = np.random.choice(tgts, size=np.random.randint(len(tgts)) + 1, replace=False).tolist()

    tgt_str = sep_token.join(tgts)
    example['target'] = tgt_str

    # print(len(tgts))
    # print(tgt_str)

    # add control prefix (number of phrases to output)
    if add_control_prefix_prob > 0.0 and np.random.rand() < add_control_prefix_prob:
        prefix_str = '<mixed><number>%d<s>' % (len(tgts))
        example['source'] = prefix_str + src_str
    else:
        example['source'] = src_str

    return example


def parse_kpdict(example, kp_concat_type, dataset_type='scipaper', sep_token='<sep>',
                 max_target_phrases=-1, max_phrase_len=-1, lowercase=False, seed=0,
                 add_control_prefix_prob=0.0):
    assert dataset_type in KP_DATASET_FIELDS
    title_field, text_field, keyword_field, category_field = KP_DATASET_FIELDS[dataset_type]

    src_str = parse_src_fn(example, title_field, text_field)
    # Ensure target is a list of phrases. Each phrase is a string, not a list of tokens
    if isinstance(example[keyword_field], str):
        example[keyword_field] = example[keyword_field].split(';')
    if len(example[keyword_field]) > 0 and isinstance(example[keyword_field][0], list):
        example[keyword_field] = [' '.join(p) for p in example[keyword_field]]
    tgt_kps = example[keyword_field]

    if max_phrase_len > 0:
        tgt_kps = [p for p in tgt_kps if len(p.split()) <= max_phrase_len]

    prefix_str = None
    if len(tgt_kps) == 0:
        tgt_str = ''
    elif kp_concat_type == 'one2one':
        # sample one tgt from multiple tgts and use it as the only tgt
        rand_idx = np.random.randint(len(tgt_kps))
        tgt_str = tgt_kps[rand_idx]
    elif kp_concat_type in KP_CONCAT_TYPES:
        # generate one2seq training data points
        src_seq = [t.text.lower() for t in spacy_nlp(src_str, disable=["textcat"])]
        tgt_seqs = [[t.text.lower() for t in spacy_nlp(p, disable=["textcat"])] for p in tgt_kps]
        order, prefix_str = obtain_sorted_indices(src_seq, tgt_seqs, sort_by=kp_concat_type, seed=seed)

        if max_target_phrases > 0 and len(order) > max_target_phrases:
            order = order[: max_target_phrases]
        tgt = [tgt_kps[idx] for idx in order]
        tgt_str = sep_token.join(tgt)
    else:
        raise NotImplementedError('Unsupported target concatenation type ' + kp_concat_type)

    if lowercase:
        src_str = src_str.lower()
        tgt_str = tgt_str.lower()

    example['source'] = src_str
    example['target'] = tgt_str

    # add control prefix (number of present/absent phrases to output)
    if prefix_str and add_control_prefix_prob > 0.0 and np.random.rand() < add_control_prefix_prob:
        example['source'] = prefix_str + src_str

    return example


def wiki_ex_parse_fn(ex_dict, sep_token,
                     max_phrase_len=8,
                     max_target_phrases=-1,
                     phrase_corr_rate=0.0,
                     random_span_rate=0.0,
                     span_len_opts=None,
                     len_distrib=None,
                     lowercase=False,
                     seed=0):
    """
    max_tgt_len=max_phrase_len*max_target_phrases + src_len*random_span_rate = 6*16+512*5%=96+25.6=121.6
    masked_word=6*8*0.1+512*5%=30.4 (30.4/512=5.9%)
    :param ex_dict:
    :param sep_token:
    :param max_phrase_len:
    :param max_target_phrases:
    :param phrase_corr_rate: replace p% * num_present_phrase present phrases from src_text with <present>
    :param random_span_rate: replace p% * num_word spans from src_text with <mask>
    :param lowercase:
    :return:
    """
    assert max_target_phrases > 0, 'max_target_phrases must be a positive integer'
    text_field = 'text'

    src_text = ex_dict[text_field]
    src_text, font_phrases, anchor_phrases = extract_phrases(src_text)

    pres_phrases = set(font_phrases + anchor_phrases)
    header_phrases = [ex_dict['title']] + ex_dict['headers']
    category_phrases = ex_dict['categories']
    seealso_phrases = ex_dict['seealso']

    if max_phrase_len:
        pres_phrases = [p for p in pres_phrases if len(p.split()) <= max_phrase_len]
        header_phrases = [p for p in header_phrases if len(p.split()) <= max_phrase_len]
        category_phrases = [p for p in category_phrases if len(p.split()) <= max_phrase_len]
        seealso_phrases = [p for p in seealso_phrases if len(p.split()) <= max_phrase_len]

    with data_utils.numpy_seed(seed):

        # present phrases
        if max_target_phrases > 0 and len(pres_phrases) > max_target_phrases / 2:
            pres_phrases = np.random.choice(pres_phrases, int(max_target_phrases / 2), replace=False).tolist()

        num_pres = len(pres_phrases)
        num_header = len(header_phrases)
        num_cat = len(category_phrases)
        num_seealso = len(seealso_phrases)

        # absent phrases
        abs_phrases = header_phrases + category_phrases + seealso_phrases
        if max_target_phrases > 0 and len(abs_phrases) > max_target_phrases / 2:
            num_cat = min(len(category_phrases), random.randint(0, int(max_target_phrases / 2 - len(header_phrases))))
            num_seealso = min(len(seealso_phrases), int(max_target_phrases / 2) - len(header_phrases) - num_cat)
            abs_phrases = header_phrases \
                          + np.random.choice(category_phrases, num_cat, replace=False).tolist()\
                          + np.random.choice(seealso_phrases, num_seealso, replace=False).tolist()

        # mask random spans
        num_infill = 0
        if random_span_rate > 0.0:
            src_tokens = src_text.split()
            num_word_left = max(1, int(random_span_rate * len(src_tokens)))

            span_lens = []
            while num_word_left > 0:
                span_len = np.random.choice(span_len_opts, p=len_distrib).tolist()
                if span_len <= num_word_left:
                    span_lens.append(span_len)
                else:
                    span_lens.append(num_word_left)
                num_word_left -= span_len
            span_lens = sorted(span_lens, reverse=True) # ensure larger spans get processed first

            spans = []
            uncovered_spans = [(0, len(src_tokens))]
            for span_len in span_lens:
                candicate_spans, noncandicate_spans = [], []
                for s in uncovered_spans:
                    if s[1] - s[0] >= span_len:
                        candicate_spans.append(s)
                    else:
                        noncandicate_spans.append(s)

                if len(candicate_spans) == 0:
                    # not possible to fit this span
                    continue
                candicate_span_id = random.choice(range(len(candicate_spans)))
                candicate_span = candicate_spans[candicate_span_id]
                candicate_span_len = candicate_span[1] - candicate_span[0]

                # sample a span start given the candidate
                span_start_offset = random.randint(0, candicate_span_len - span_len + 1)
                span_left = candicate_span[0] + span_start_offset
                spans.append((span_left, span_left + span_len))

                # maintain the new candidate lists
                if span_start_offset == 0:
                    leftover_spans = [(candicate_span[0] + span_len, candicate_span[1] + 1)]
                elif span_start_offset == candicate_span_len - span_len:
                    leftover_spans = [(candicate_span[0], candicate_span[1] - span_len)]
                else:
                    leftover_spans = [(candicate_span[0], span_left), (span_left + span_len, candicate_span[1] + 1)]

                uncovered_spans = noncandicate_spans + leftover_spans

            spans = sorted(spans, key=lambda x: x[0], reverse=False)
            masked_src_tokens = []
            prev_span_end = 0
            for s in spans:
                masked_src_tokens.extend(src_tokens[prev_span_end: s[0]])
                masked_src_tokens.append('<infill>')
                prev_span_end = s[1]
            masked_src_tokens.extend(src_tokens[prev_span_end:])

            infill_phrases = [' '.join(src_tokens[s[0]: s[1]]) for s in spans]
            num_infill = len(infill_phrases)
            src_text = ' '.join(masked_src_tokens)

        # mask random present phrases
        if phrase_corr_rate > 0.0 and len(pres_phrases) > 0:
            num_mask_kp = min(1, int(len(pres_phrases) * phrase_corr_rate))
            mask_pres_phrases = np.random.choice(pres_phrases, num_mask_kp, replace=False).tolist()
            for p in mask_pres_phrases:
                src_text = re.sub(p, '<present>', src_text, flags=re.IGNORECASE)

    prefix_str = '<present>%d<header>%d<category>%d<seealso>%d<infill>%d<s>' \
                 % (num_pres, num_header, num_cat, num_seealso, num_infill)

    src_text = prefix_str + src_text
    tgt_text = sep_token.join(pres_phrases + abs_phrases + infill_phrases)

    if lowercase:
        return src_text.lower(), tgt_text.lower()
    return {
            'id': ex_dict['id'],
            'source': src_text,
            'target': tgt_text
    }


def obtain_sorted_indices(src, tgt_seqs, sort_by, seed):
    """
    :param src: used for verbatim and alphabetical
    :param tgt_seqs:
    :param sort_by:
    :return:
    """
    num_tgt = len(tgt_seqs)
    prefix_str = None

    if sort_by == 'random':
        with data_utils.numpy_seed(seed):
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

        # add control prefix (number of phrases to output)
        prefix_str = '<present>%d<absent>%d<s>' % (len(present_tgt_idx), len(absent_tgt_idx))
    else:
        raise NotImplementedError('Unsupported sort_by value: ' + sort_by)

    if sort_by.endswith('reverse'):
        sorted_id = sorted_id[::-1]

    return np.asarray(sorted_id, dtype=int), prefix_str


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


def findBalanced(text, openDelim=['[['], closeDelim=[']]']):
    """
    Assuming that text contains a properly balanced expression using
    :param openDelim: as opening delimiters and
    :param closeDelim: as closing delimiters.
    :return: an iterator producing pairs (start, end) of start and end
    positions in text containing a balanced expression.
    """
    openPat = '|'.join([re.escape(x) for x in openDelim])
    # pattern for delimiters expected after each opening delimiter
    afterPat = {o: re.compile(openPat + '|' + c, re.DOTALL) for o, c in zip(openDelim, closeDelim)}
    stack = []
    start = 0
    cur = 0
    # end = len(text)
    startSet = False
    startPat = re.compile(openPat)
    nextPat = startPat
    while True:
        next = nextPat.search(text, cur)
        if not next:
            return
        if not startSet:
            start = next.start()
            startSet = True
        delim = next.group(0)
        if delim in openDelim:
            stack.append(delim)
            nextPat = afterPat[delim]
        else:
            opening = stack.pop()
            # assert opening == openDelim[closeDelim.index(next.group(0))]
            if stack:
                nextPat = afterPat[stack[-1]]
            else:
                yield start, next.end()
                nextPat = startPat
                start = next.end()
                startSet = False
        cur = next.end()


def replaceInternalLinks(text, return_anchor_text=False):
    """
    Replaces internal links of the form:
    [[title |...|label]]trail

    with title concatenated with trail, when present, e.g. 's' for plural.

    See https://www.mediawiki.org/wiki/Help:Links#Internal_links
    """
    # call this after removal of external links, so we need not worry about
    # triple closing ]]].
    cur = 0
    res = ''
    phrase_list = []
    for s, e in findBalanced(text):
        m = tailRE.match(text, e)
        if m:
            trail = m.group(0)
            end = m.end()
        else:
            trail = ''
            end = e
        inner = text[s + 2:e - 2]
        # find first |
        pipe = inner.find('|')
        if pipe < 0:
            title = inner
            label = title
        else:
            title = inner[:pipe].rstrip()
            # find last |
            curp = pipe + 1
            for s1, e1 in findBalanced(inner):
                last = inner.rfind('|', curp, s1)
                if last >= 0:
                    pipe = last  # advance
                curp = e1
            label = inner[pipe + 1:].strip()

        # phrase_list.append(title.strip())
        phrase_list.append(label.strip())
        res += text[cur:s] + label + trail
        cur = end
    if return_anchor_text:
        return res + text[cur:], phrase_list
    else:
        return res + text[cur:]


bold_italic = re.compile(r"'''''(.*?)'''''")
bold = re.compile(r"'''(.*?)'''")
italic_quote = re.compile(r"''\"([^\"]*?)\"''")
italic = re.compile(r"''(.*?)''")
quote_quote = re.compile(r'""([^"]*?)""')
tailRE = re.compile('\w+')

def extract_phrases(text):
    # Extract bold/italic text and internal links

    # Extract bold/anchor texts
    font_phrases = bold_italic.findall(text)
    font_phrases += bold.findall(text)
    font_phrases += italic_quote.findall(text)
    font_phrases += italic.findall(text)
    font_phrases += quote_quote.findall(text)
    font_phrases = [p.strip('\',\"') for p in font_phrases]
    font_phrases = list(set(font_phrases))

    # Handle bold/italic/quote
    text = bold_italic.sub(r'\1', text)
    text = bold.sub(r'\1', text)
    text = italic_quote.sub(r'"\1"', text)
    text = italic.sub(r'"\1"', text)
    text = quote_quote.sub(r'"\1"', text)
    # replace internal links
    text, anchor_phrases = replaceInternalLinks(text, return_anchor_text=True)
    anchor_phrases = [p.strip('\',\"') for p in anchor_phrases]
    anchor_phrases = list(set(anchor_phrases))

    return text, font_phrases, anchor_phrases
