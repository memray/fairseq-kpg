# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import json
import logging
import os
from functools import partial

from fairseq import utils
from fairseq.data.encoders.hf_bpe import HuggingFacePretrainedBPE
from fairseq.data.keyphrase_pair_dataset import KeyphrasePairDataset
from fairseq.data.keyphrase_raw_dataset import KeyphraseRawDataset

from . import register_task
from fairseq.tasks import LegacyFairseqTask
from fairseq.tasks.keyphrasification_utils import KP_DATASET_FIELDS, KP_CONCAT_TYPES, parse_kpdict, maybe_replace_target

logger = logging.getLogger(__name__)


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
        parser.add_argument('--label-data', type=str, help='colon separated path to file list, \
                            all files should have the same number of lines as `data`. \
                            If given, target labels will be sampled from those labels.')
        parser.add_argument('--label-sample-ratio', type=str, help='Sampling proportion of labels from each label file.')
        parser.add_argument("--dict-path", type=str,
                            help='path to vocab.bpe.')
        parser.add_argument('--add-control-prefix-prob', type=float, default=0.0,
                            help='Roll to decide whether adding a prefix to indicate number of phrases to predict.'
                                 '1.0 always add prefix, 0.0 always not add prefix.')
        parser.add_argument("--kp-concat-type", choices=KP_CONCAT_TYPES,
                            help='how to present target sequence')
        parser.add_argument("--dataset-type", choices=list(KP_DATASET_FIELDS.keys()), required=True,
                            help='Specify type of dataset, select from ' + str(list(KP_DATASET_FIELDS.keys())))
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
        else:
            raise NotImplementedError('Unsupported tokenizer %s' % args.tokenizer)

        # Load dictionaries, see https://github.com/pytorch/fairseq/issues/1432
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
        assert len(paths) == 1, 'Currently only one dataset is supported for keyphrase task.'
        root_path = paths[0]
        if not os.path.exists(root_path):
            raise FileNotFoundError('Dataset not found: {} ({})'.format(split, root_path))
        if os.path.isdir(root_path):
            data_names = sorted(os.listdir(root_path))
            data_file_name = data_names[epoch % len(data_names)]
            data_path = os.path.join(root_path, data_file_name)
            is_folder = True
        else:
            data_path = root_path
            is_folder = False

        dataset = KeyphraseRawDataset(data_path, self.args.dataset_type)
        logger.info('{} {} {} examples'.format(
            data_path, split, len(dataset)
        ))

        if self.args.label_data:
            self.args.label_sample_ratio = eval(self.args.label_sample_ratio)
            assert sum(self.args.label_sample_ratio) == 1.0
            label_paths = utils.split_paths(self.args.label_data)
            for labelset_id, labelset_path in enumerate(label_paths):
                if is_folder:
                    labelset_path = os.path.join(labelset_path, data_file_name)
                    assert os.path.exists(labelset_path), 'labelset data does not exist, path: '+ labelset_path
                if labelset_path.startswith('__'):
                    # dynamic labels like random span will be added in preprocessing
                    [data_ex.update({'target%d' % labelset_id: labelset_path})
                     for data_ex in dataset.example_dicts]
                else:
                    # load extra labels from disk
                    label_exs = [json.loads(l) for l in open(labelset_path, 'r')]
                    assert len(label_exs) == len(dataset), \
                        'Size of additional label data (#=%d) must match the size of dataset (#=%d).' % (len(label_exs), len(dataset))
                    # model's outputs may tokenized, concatenate them to strings
                    if not isinstance(label_exs[0]['pred_sents'][0], str):
                        for label_ex in label_exs:
                            label_ex['pred_sents'] = [' '.join(p) for p in label_ex['pred_sents']]
                    [data_ex.update({'target%d' % labelset_id: label_ex['pred_sents']})
                        for data_ex, label_ex in zip(dataset.example_dicts, label_exs)]
                    del label_exs
        else:
            self.args.label_sample_ratio = None

        # dataset.example_dicts = dataset.example_dicts[2800:]

        # configure transform functions
        kp_parse_fn = partial(parse_kpdict,
                              kp_concat_type=self.args.kp_concat_type, dataset_type=self.args.dataset_type,
                              sep_token=self.text_tokenizer.sep_token,
                              max_target_phrases=self.args.max_target_phrases,
                              max_phrase_len=self.args.max_phrase_len,
                              lowercase=self.args.lowercase,
                              seed=self.args.seed + epoch if split != 'test' else 0)

        target_replace_fn = partial(maybe_replace_target,
                                    label_sample_ratio=self.args.label_sample_ratio,
                                    max_target_phrases=self.args.max_target_phrases,
                                    max_phrase_len=self.args.max_phrase_len,
                                    add_control_prefix_prob=self.args.add_control_prefix_prob,
                                    fix_target_number=False, allow_duplicate=False,
                                    sep_token=self.text_tokenizer.sep_token,
                                    seed=self.args.seed + epoch if split != 'test' else 0
                                    )

        transform_fns = [kp_parse_fn, target_replace_fn]

        self.datasets[split] = KeyphrasePairDataset(
            dataset, vocab=self.dictionary, sizes=dataset.sizes,
            text_tokenizer=self.text_tokenizer, transform_fns=transform_fns,
            left_pad_source=self.args.left_pad_source,
            left_pad_target=self.args.left_pad_target,
            max_source_length=self.args.max_source_length,
            max_target_length=self.args.max_target_length,
            # shuffle=False,
            shuffle=(split != 'test'),
            sort_by_length=True
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
