# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import logging
import os
from functools import partial

import numpy as np

from fairseq import metrics, utils
from fairseq.data import (
    ConcatDataset,
)
from fairseq.data.encoders.hf_bpe import HuggingFacePretrainedBPE
from fairseq.data.keyphrase_pair_dataset import KeyphrasePairDataset
from fairseq.data.keyphrase_raw_dataset import KeyphraseRawDataset

from . import register_task
from fairseq.tasks import LegacyFairseqTask
from fairseq.tasks.keyphrasification_utils import KP_DATASET_FIELDS, KP_CONCAT_TYPES, kpdict_parse_fn

logger = logging.getLogger(__name__)


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
        parser.add_argument("--valid-data", type=str,
                            help='directory of valid data.')
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

        parser.add_argument('--max-phrase-len', default=8, type=int, metavar='N',
                            help='filter short phrases. used in pretrain code.')
        parser.add_argument('--max-target-phrases', default=-1, type=int, metavar='N',
                            help='max number of phrases in target. If exceeds, random max_num phrases will be used.'
                                 'If -1, all phrases will be retained.')

        parser.add_argument('--max-source-positions', default=1024, type=int, metavar='N',
                            help='(for initializing model embeddings) max number of tokens in the source sequence')
        parser.add_argument('--max-target-positions', default=1024, type=int, metavar='N',
                            help='(initializing model embeddings) max number of tokens in the target sequence')
        parser.add_argument('--upsample-primary', default=1, type=int,
                            help='amount to upsample primary dataset')
        parser.add_argument('--lowercase', default='False', type=str, metavar='BOOL',
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
        args.lowercase = utils.eval_bool(args.lowercase)

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
