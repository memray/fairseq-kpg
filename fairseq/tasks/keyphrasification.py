# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from argparse import Namespace
import json
import itertools
import logging
import os
from functools import partial

from fairseq import options
import numpy as np

from fairseq.data.encoders.gpt2_bpe import get_encoder

from fairseq import metrics, utils
from fairseq.data import (
    ConcatDataset,
)
from fairseq.data.encoders.hf_bpe import HuggingFacePretrainedBPE
from fairseq.data.keyphrase_pair_dataset import KeyphrasePairDataset
from fairseq.data.keyphrase_raw_dataset import KeyphraseRawDataset

from fairseq.tasks import register_task, LegacyFairseqTask


KP_CONCAT_TYPES = ['one2one', 'random',
                   'pres_abs', 'abs_pres',
                   'nosort', 'nosort_reverse',
                   'alphab', 'alphab_reverse',
                   'length', 'length_reverse']

KP_DATASET_FIELDS = {'scipaper': ('title', 'abstract', 'keywords', None),
                     'qa': ('title', 'question', 'tags', 'categories'),
                     'webpage': ('url', 'text', 'KeyPhrases', None),
                     'news': ('title', 'abstract', 'keyword', 'categories')}

logger = logging.getLogger(__name__)


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


def load_kppair_dataset(
    data_path, split,
    text_tokenizer, dictionary,
    kp_concat_type,
    combine, upsample_primary,
    left_pad_source, left_pad_target,
    max_source_length, max_target_length,
    max_target_phrases,
    num_buckets=0,
    shuffle=True,
    pad_to_multiple=1,
    lowercase=False,
    dataset_type=None
):
    src_datasets = []
    tgt_datasets = []

    for k in itertools.count():
        split_k = split + ('-'+str(k) if split == 'train' else '')
        filepath = os.path.join(os.path.realpath(data_path), '{}.json'.format(split_k))

        # for cases that train set is not split into pieces
        if not os.path.exists(filepath):
            filepath = os.path.join(os.path.realpath(data_path), '{}.json'.format(split))
            combine = False

        if not os.path.exists(filepath):
            if k > 0:
                break
            else:
                raise FileNotFoundError('Dataset not found: {} ({})'.format(split, data_path))

        src_dataset = KeyphraseRawDataset(filepath, dataset_type)
        src_datasets.append(src_dataset)

        # tgt_dataset = KeyphraseRawDataset(ex_dicts)
                                    # parse_fn=partial(parse_tgt_fn, kp_concat_type=tgt_type, tokenizer=text_tokenizer.tokenizer),
                                    # text_tokenizer=text_tokenizer)
        # if tgt_dataset is not None:
        #     tgt_datasets.append(tgt_dataset)

        logger.info('{} {} {} examples'.format(
            data_path, split_k, len(src_dataset)
        ))

        if not combine:
            break

    assert len(src_datasets) == len(tgt_datasets) or len(tgt_datasets) == 0

    if len(src_datasets) == 1:
        src_dataset = src_datasets[0]
        # tgt_dataset = tgt_datasets[0] if len(tgt_datasets) > 0 else None
    else:
        sample_ratios = [1] * len(src_datasets)
        sample_ratios[0] = upsample_primary
        src_dataset = ConcatDataset(src_datasets, sample_ratios)
        # if len(tgt_datasets) > 0:
        #     tgt_dataset = ConcatDataset(tgt_datasets, sample_ratios)
        # else:
        #     tgt_dataset = None

    # if prepend_bos:
    #     assert hasattr(dictionary, "bos_index") and hasattr(dictionary, "bos_index")
    #     src_dataset = PrependTokenDataset(src_dataset, dictionary.bos())
    #     if tgt_dataset is not None:
    #         tgt_dataset = PrependTokenDataset(tgt_dataset, dictionary.bos())

    if not dataset_type:
        dataset_type = src_dataset.dataset_type
    parse_fn = partial(kpdict_parse_fn, tokenizer=text_tokenizer,
                       kp_concat_type=kp_concat_type, dataset_type=dataset_type,
                       max_target_phrases=max_target_phrases, lowercase=lowercase)
    return KeyphrasePairDataset(
        src_dataset, src_dict=dictionary, src_sizes=src_dataset.sizes,
        text_tokenizer=text_tokenizer, parse_fn=parse_fn,
        # tgt=tgt_dataset, tgt_sizes=tgt_dataset.sizes, tgt_dict=dictionary,
        kp_concat_type=kp_concat_type,
        shuffle=shuffle,
        left_pad_source=left_pad_source,
        left_pad_target=left_pad_target,
        max_source_length=max_source_length,
        max_target_length=max_target_length,
        num_buckets=num_buckets,
        pad_to_multiple=pad_to_multiple,
    )


@register_task('keyphrasification')
class KeyphrasificationTask(LegacyFairseqTask):
    """
    Translate from one (source) language to another (target) language.

    Args:
        src_dict (~fairseq.data.Dictionary): dictionary for the source language
        tgt_dict (~fairseq.data.Dictionary): dictionary for the target language

    .. note::

        The translation task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate` and :mod:`fairseq-interactive`.

    The translation task provides the following additional command-line
    arguments:

    .. argparse::
        :ref: fairseq.tasks.translation_parser
        :prog:
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('data', help='colon separated path to data directories list, \
                            will be iterated upon during epochs in round-robin manner; \
                            however, valid and test data are always in the first directory to \
                            avoid the need for repeating them in all directories')
        parser.add_argument("--dict-path", type=str,
                            help='path to vocab.bpe.')
        parser.add_argument("--kp-concat-type", default='nosort',
                            choices=KP_CONCAT_TYPES,
                            help='how to present target sequence')
        parser.add_argument("--num-encoder-workers", type=int, default=20)
        parser.add_argument('--left-pad-source', default='False', type=str, metavar='BOOL',
                            help='pad the source on the left')
        parser.add_argument('--left-pad-target', default='False', type=str, metavar='BOOL',
                            help='pad the target on the left')
        parser.add_argument('--max-source-length', default=512, type=int, metavar='N',
                            help='(for processing data) max number of tokens in source')
        parser.add_argument('--max-target-length', default=128, type=int, metavar='N',
                            help='(for processing data) max number of tokens in target')
        parser.add_argument('--max-target-phrases', default=-1, type=int, metavar='N',
                            help='max number of phrases in target. If exceeds, random max_num phrases will be used.'
                                 'If -1, all phrases will be retained.')
        parser.add_argument('--max-source-positions', default=1024, type=int, metavar='N',
                            help='(for initializing model embeddings) max number of tokens in the source sequence')
        parser.add_argument('--max-target-positions', default=1024, type=int, metavar='N',
                            help='(initializing model embeddings) max number of tokens in the target sequence')
        parser.add_argument('--upsample-primary', default=1, type=int,
                            help='amount to upsample primary dataset')
        parser.add_argument('--lowercase', default='True', type=str, metavar='BOOL',
                            help='lowercase all texts (uncased)')
        parser.add_argument('--num-batch-buckets', default=0, type=int, metavar='N',
                            help='if >0, then bucket source and target lengths into N '
                                 'buckets and pad accordingly; this is useful on TPUs '
                                 'to minimize the number of compilations')
        # fmt: on

    def __init__(self, args):
        super().__init__(args)
        self.tokenizer = args.tokenizer

        assert os.path.exists(args.bpe_vocab) and os.path.exists(args.bpe_merges),\
            "Both vocab and merges are needed to load Huggingface tokenizer"
        assert os.path.exists(args.dict_path),\
            "Fairseq dict file is needed."
        if args.bpe == 'hf_pretrained_bpe':
            text_tokenizer = HuggingFacePretrainedBPE.load(args)
            logger.info('Loaded dictionary from Huggingface {}'.format(args.bpe_vocab))
        # elif args.bpe == 'gpt2':
        #     text_tokenizer = get_encoder(args.encoder_json, args.vocab_bpe)
        #     logger.info('Loaded dictionary from fairseq GPT2')
        else:
            raise NotImplementedError('Unsupported tokenizer %s' % args.tokenizer)

        # Lsoad dictionaries, see https://github.com/pytorch/fairseq/issues/1432
        dictionary = self.load_dictionary(args.dict_path)
        # Note that in vocab.txt, madeupword0001 is replaced with <sep>.
        #   It seems other special tokens like <present> don't need explicit setting.
        # dictionary.indices[text_tokenizer.mask_token] = text_tokenizer.mask_token_id
        # setattr(dictionary, 'mask_word', text_tokenizer.mask_token)
        # setattr(dictionary, 'mask_index', text_tokenizer.mask_token_id)
        # dictionary.indices[text_tokenizer.sep_token] = text_tokenizer.sep_token_id
        # setattr(dictionary, 'sep_word', text_tokenizer.sep_token)
        # setattr(dictionary, 'sep_index', text_tokenizer.sep_token_id)
        # dictionary.nspecial = len(text_tokenizer.all_special_tokens)
        # embedding initilization replies on symbols (or manually edit the dict file to add SEP and MASK)
        # dictionary.symbols.extend([text_tokenizer.mask_token, text_tokenizer.sep_token])

        self.dictionary = dictionary
        self.text_tokenizer = text_tokenizer
        assert len(dictionary.indices) == text_tokenizer.vocab_size
        self.kp_concat_type = args.kp_concat_type

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        args.left_pad_source = utils.eval_bool(args.left_pad_source)
        args.left_pad_target = utils.eval_bool(args.left_pad_target)

        paths = utils.split_paths(args.data)
        assert len(paths) > 0

        return cls(args)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = utils.split_paths(self.args.data)
        assert len(paths) > 0
        if split != getattr(self.args, "train_subset", None):
            # if not training data set, use the first shard for valid and test
            paths = paths[:1]
        data_path = paths[(epoch - 1) % len(paths)]

        self.datasets[split] = load_kppair_dataset(
            data_path, split,
            self.text_tokenizer, self.dictionary,
            self.kp_concat_type,
            combine=combine,
            upsample_primary=self.args.upsample_primary,
            left_pad_source=self.args.left_pad_source,
            left_pad_target=self.args.left_pad_target,
            max_source_length=self.args.max_source_length,
            max_target_length=self.args.max_target_length,
            max_target_phrases=self.args.max_target_phrases,
            num_buckets=self.args.num_batch_buckets,
            shuffle=(split != 'test'),
            pad_to_multiple=self.args.required_seq_len_multiple,
            lowercase=self.args.lowercase
        )

    def build_dataset_for_inference(self, src_tokens, src_lengths, constraints=None):
        return KeyphrasePairDataset(src_tokens, src_lengths, self.source_dictionary,
                                   tgt_dict=self.target_dictionary,
                                   constraints=constraints)

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        return (self.args.max_source_positions, self.args.max_target_positions)

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        return self.dictionary

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.dictionary
