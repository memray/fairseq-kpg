# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import random
import re

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


def wiki_ex_parse_fn(ex_dict, sep_token,
                     max_phrase_len=8,
                     max_target_phrases=-1,
                     phrase_corr_rate=0.0,
                     random_span_rate=0.0,
                     span_len_opts=None,
                     len_distrib=None,
                     lowercase=False):
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

    # present phrases
    if max_target_phrases > 0 and len(pres_phrases) > max_target_phrases / 2:
        pres_phrases = random.sample(pres_phrases, int(max_target_phrases / 2))

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
                      + random.sample(category_phrases, num_cat) \
                      + random.sample(seealso_phrases, num_seealso)

    # mask random spans
    num_infill = 0
    if random_span_rate > 0.0:
        src_tokens = src_text.split()
        num_word_left = max(1, int(random_span_rate * len(src_tokens)))

        span_lens = []
        while num_word_left > 0:
            span_len = np.random.choice(span_len_opts, p=len_distrib)
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
        mask_pres_phrases = random.sample(pres_phrases, num_mask_kp)
        for p in mask_pres_phrases:
            src_text = re.sub(p, '<present>', src_text, flags=re.IGNORECASE)

    prefix_str = '<present>%d<header>%d<category>%d<seealso>%d<infill>%d<s>' \
                 % (num_pres, num_header, num_cat, num_seealso, num_infill)

    src_text = prefix_str + src_text
    tgt_text = sep_token.join(pres_phrases + abs_phrases + infill_phrases)

    if lowercase:
        return src_text.lower(), tgt_text.lower()
    return src_text, tgt_text


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
