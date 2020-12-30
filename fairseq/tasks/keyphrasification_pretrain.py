# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
import os
import re
from functools import partial

from fairseq import metrics, utils
from fairseq.data import ConcatDataset
from fairseq.data.keyphrase_pair_dataset import KeyphrasePairDataset
from fairseq.data.keyphrase_raw_dataset import KeyphraseRawDataset

from fairseq.tasks import register_task, LegacyFairseqTask
from fairseq.tasks.keyphrasification import KeyphrasificationTask, load_kppair_dataset
from fairseq.tasks.keyphrasification_utils import parse_src_fn, KP_DATASET_FIELDS, KP_CONCAT_TYPES, \
    obtain_sorted_indices
from fairseq.utils import logger


@register_task('keyphrasification_pretrain')
class KeyphrasificationPretrainTask(KeyphrasificationTask):
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
        KeyphrasificationTask.add_args(parser)
        parser.add_argument("--source-only", default='False', type=str, metavar='BOOL',
                            help='.')
        # fmt: on

    def __init__(self, args):
        super().__init__(args)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        if split == 'train':
            # Pretraining data such as wiki is stored in folders (AA,AB,AC etc.), each contains multiple files (shards)
            # in each epoch we only load one folder
            paths = utils.split_paths(self.args.data)
            paths = sorted([os.path.join(self.args.data, subdir) for path in paths for subdir in os.listdir(path)])
            paths = [paths[(epoch - 1) % len(paths)]]
        elif split == 'valid':
            paths = utils.split_paths(self.args.valid_data)
        else:
            logger.error('Invalid split name: %s' % split)


        self.datasets[split] = load_pretrain_dataset(
            paths, split,
            self.text_tokenizer, self.dictionary,
            upsample_primary=self.args.upsample_primary,
            left_pad_source=self.args.left_pad_source,
            left_pad_target=self.args.left_pad_target,
            max_source_length=self.args.max_source_length,
            max_target_length=self.args.max_target_length,
            max_target_phrases=self.args.max_target_phrases,
            num_buckets=self.args.num_batch_buckets,
            shuffle=(split.startswith('train')),
            pad_to_multiple=self.args.required_seq_len_multiple,
            lowercase=self.args.lowercase
        )


def wiki_ex_parse_fn(ex_dict, tokenizer, max_target_phrases=-1, lowercase=False):
    title_field, text_field = '', 'text'

    src_str = ex_dict[text_field]
    tgt_str = ex_dict[text_field]

    if lowercase:
        return src_str.lower(), tgt_str.lower()
    return src_str, tgt_str


def load_pretrain_dataset(
        data_paths, split,
        text_tokenizer, dictionary,
        upsample_primary,
        left_pad_source, left_pad_target,
        max_source_length, max_target_length,
        max_target_phrases,
        num_buckets=0,
        shuffle=True,
        pad_to_multiple=1,
        lowercase=False,
        dataset_type=None
):
    data_files = []
    # TODO for now it only supports wiki yet
    file_pattern = 'wiki_\d+'

    for data_path in sorted(data_paths):
        _data_files = []
        for root, dirs, files in os.walk(data_path):
            for file in files:
                if not re.match(file_pattern, file):
                    continue

                filepath = os.path.join(root, file)
                data_files.append(filepath)

        logger.info('Find {} shards at {}'.format(len(_data_files), data_path))
        data_files.extend(_data_files)

    data_files = sorted(data_files) # to ensure the consistent data order, will be sampled later
    raw_datasets = [KeyphraseRawDataset(filepath, dataset_type) for filepath in data_files]
    logger.info('[SPLIT-{}]: load {} shards and {} data examples in total.'.format(
        split, len(raw_datasets), sum([len(ds) for ds in raw_datasets])))

    if len(raw_datasets) == 1:
        raw_dataset = raw_datasets[0]
        # tgt_dataset = tgt_datasets[0] if len(tgt_datasets) > 0 else None
    else:
        sample_ratios = [1] * len(raw_datasets)
        sample_ratios[0] = upsample_primary
        raw_dataset = ConcatDataset(raw_datasets, sample_ratios)

    parse_fn = partial(wiki_ex_parse_fn, tokenizer=text_tokenizer,
                       max_target_phrases=max_target_phrases, lowercase=lowercase)

    return KeyphrasePairDataset(
        raw_dataset, src_dict=dictionary, src_sizes=raw_dataset.sizes,
        text_tokenizer=text_tokenizer, parse_fn=parse_fn,
        shuffle=shuffle,
        left_pad_source=left_pad_source,
        left_pad_target=left_pad_target,
        max_source_length=max_source_length,
        max_target_length=max_target_length,
        num_buckets=num_buckets,
        pad_to_multiple=pad_to_multiple,
    )