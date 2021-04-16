# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import itertools
import logging
import os
import re

import numpy as np
from fairseq.data.encoders.hf_bpe import HuggingFacePretrainedBPE

from fairseq import utils
from fairseq.data import (
    IdDataset,
    NestedDictionaryDataset,
    NumelDataset,
    NumSamplesDataset,
    PrependTokenDataset,
    RightPadDataset,
    SortDataset,
    TokenBlockDataset,
    data_utils,
    ConcatDataset)
from fairseq.data.mlm_otf_dataset import MlmOtfDataset
from fairseq.data.raw_text_dataset import RawTextDataset
from fairseq.tasks import LegacyFairseqTask, register_task


logger = logging.getLogger(__name__)



@register_task("mlm_otf")
class MlmOtfTask(LegacyFairseqTask):
    """Task for training masked language models (e.g., BERT, RoBERTa)."""

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument(
            "data",
            help="colon separated path to data directories list, \
                            will be iterated upon during epochs in round-robin manner",
        )
        parser.add_argument("--valid-data", type=str,
                            help='directory of valid data.')
        parser.add_argument("--no-bos-eos", action="store_true",
                            help='disable adding bos/eos to the input sequence.')
        parser.add_argument("--max-shards-per-epoch", type=int, default=20,
                            help='how many shards to be loaded in the memory.')
        parser.add_argument("--dict-path", type=str,
                            help='path to vocab.bpe.')
        parser.add_argument("--text-field", type=str, default=None,
                            help='if given, treat input file in jsonl format and take the data of this text field as input.')
        parser.add_argument("--num-encoder-workers", type=int, default=20)

        parser.add_argument(
            "--tokens-per-sample",
            default=512,
            type=int,
            help="max number of total tokens over all segments "
            "per sample for BERT dataset",
        )
        parser.add_argument(
            "--mask-prob",
            default=0.15,
            type=float,
            help="probability of replacing a token with mask",
        )
        parser.add_argument(
            "--leave-unmasked-prob",
            default=0.1,
            type=float,
            help="probability that a masked token is unmasked",
        )
        parser.add_argument(
            "--random-token-prob",
            default=0.1,
            type=float,
            help="probability of replacing a token with a random token",
        )


    def __init__(self, args):
        super().__init__(args)
        self.text_field = args.text_field
        self.no_bos_eos = args.no_bos_eos
        self.seed = args.seed

        self.tokens_per_sample = args.tokens_per_sample
        self.mask_prob = args.mask_prob
        self.leave_unmasked_prob = args.leave_unmasked_prob
        self.random_token_prob = args.random_token_prob

        assert os.path.exists(args.bpe_vocab) and os.path.exists(args.bpe_merges),\
            "Both vocab and merges are needed to load Huggingface tokenizer"
        assert os.path.exists(args.dict_path),\
            "Fairseq dict file is needed."
        text_tokenizer = HuggingFacePretrainedBPE.load(args)
        logger.info('Loaded dictionary from Huggingface {}'.format(args.bpe_vocab))
        # Load dictionaries, see https://github.com/pytorch/fairseq/issues/1432
        dictionary = self.load_dictionary(args.dict_path)

        self.dictionary = dictionary
        self.text_tokenizer = text_tokenizer
        assert len(dictionary.indices) == text_tokenizer.vocab_size

        # add mask token
        self.mask_idx = text_tokenizer.mask_token_id


    @classmethod
    def setup_task(cls, args, **kwargs):
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

        if split == 'train':
            # Pretraining data such as wiki is stored in folders (AA,AB,AC etc.), each contains multiple files (shards)
            # in each epoch we only load one folder
            paths = utils.split_paths(self.args.data)
            subdir_paths = sorted([os.path.join(path, subdir) for path in paths for subdir in os.listdir(path)])
            subdir_paths = [subdir_paths[(epoch - 1) % len(subdir_paths)]]
        elif split == 'valid':
            subdir_paths = utils.split_paths(self.args.valid_data)
        else:
            logger.error('Invalid split name: %s' % split)

        data_files = []
        # TODO for now it only supports wiki/book yet
        file_pattern = r'wiki_\d+|book_\d+\.json'

        for subdir_path in sorted(subdir_paths):
            _data_files = []
            for root, dirs, files in os.walk(subdir_path):
                for file in files:
                    if not re.match(file_pattern, file):
                        continue

                    filepath = os.path.join(root, file)
                    _data_files.append(filepath)

            logger.info('Find {} shards at {}'.format(len(_data_files), subdir_path))
            data_files.extend(_data_files)

        data_files = sorted(data_files)
        raw_datasets = [RawTextDataset(filepath, text_field=self.text_field) for filepath in data_files]
        logger.info('[SPLIT-{}]: load {} shards and {} data examples at epoch {}.'.format(
            split, len(raw_datasets), sum([len(ds) for ds in raw_datasets]), epoch))

        if len(raw_datasets) == 1:
            dataset = raw_datasets[0]
        else:
            dataset = ConcatDataset(raw_datasets)

        self.datasets[split] = MlmOtfDataset(
            dataset, data_sizes=dataset.sizes,
            vocab=self.dictionary,
            text_tokenizer=self.text_tokenizer,
            shuffle=(split.startswith('train')),
            epoch=epoch,
            seed=self.seed,
            tokens_per_sample=self.tokens_per_sample,
            mask_prob=self.mask_prob,
            leave_unmasked_prob=self.leave_unmasked_prob,
            random_token_prob=self.random_token_prob,
            split=split,
            no_bos_eos=self.no_bos_eos
        )


    def build_dataset_for_inference(self, src_tokens, src_lengths, sort=True):
        src_dataset = RightPadDataset(
            TokenBlockDataset(
                src_tokens,
                src_lengths,
                self.args.tokens_per_sample - 1,  # one less for <s>
                pad=self.source_dictionary.pad(),
                eos=self.source_dictionary.eos(),
                break_mode="eos",
            ),
            pad_idx=self.source_dictionary.pad(),
        )
        src_dataset = PrependTokenDataset(src_dataset, self.source_dictionary.bos())
        src_dataset = NestedDictionaryDataset(
            {
                "id": IdDataset(),
                "net_input": {
                    "src_tokens": src_dataset,
                    "src_lengths": NumelDataset(src_dataset, reduce=False),
                },
            },
            sizes=src_lengths,
        )
        if sort:
            src_dataset = SortDataset(src_dataset, sort_order=[src_lengths])
        return src_dataset

    @property
    def source_dictionary(self):
        return self.dictionary

    @property
    def target_dictionary(self):
        return self.dictionary
