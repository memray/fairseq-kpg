# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

import numpy as np
import torch

from fairseq.data import data_utils, FairseqDataset


logger = logging.getLogger(__name__)


class KeyphrasePairDataset(FairseqDataset):
    """
    A pair of torch.utils.data.Datasets.

    Args:
        src (torch.utils.data.Dataset): source dataset to wrap
        src_sizes (List[int]): source sentence lengths
        src_dict (~fairseq.data.Dictionary): source vocabulary
        tgt (torch.utils.data.Dataset, optional): target dataset to wrap
        tgt_sizes (List[int], optional): target sentence lengths
        tgt_dict (~fairseq.data.Dictionary, optional): target vocabulary
        left_pad_source (bool, optional): pad source tensors on the left side
            (default: True).
        left_pad_target (bool, optional): pad target tensors on the left side
            (default: False).
        shuffle (bool, optional): shuffle dataset elements before batching
            (default: True).
        input_feeding (bool, optional): create a shifted version of the targets
            to be passed into the model for teacher forcing (default: True).
        remove_eos_from_source (bool, optional): if set, removes eos from end
            of source if it's present (default: False).
        append_eos_to_target (bool, optional): if set, appends eos to end of
            target if it's absent (default: False).
        align_dataset (torch.utils.data.Dataset, optional): dataset
            containing alignments.
        constraints (Tensor, optional): 2d tensor with a concatenated, zero-
            delimited list of constraints for each sentence.
        append_bos (bool, optional): if set, appends bos to the beginning of
            source/target sentence.
        num_buckets (int, optional): if set to a value greater than 0, then
            batches will be bucketed into the given number of batch shapes.
    """

    def __init__(
        self, src, src_dict, src_sizes,
        text_tokenizer, parse_fn,
        tgt=None, tgt_dict=None, tgt_sizes=None,
        shuffle=True, input_feeding=True,
        left_pad_source=False, left_pad_target=False,
        max_source_length=None, max_target_length=None,
        num_buckets=0,
        pad_to_multiple=1,
    ):
        if tgt_dict is not None:
            assert src_dict.pad() == tgt_dict.pad()
            assert src_dict.eos() == tgt_dict.eos()
            assert src_dict.unk() == tgt_dict.unk()
        if tgt is not None:
            assert len(src) == len(tgt), "Source and target must contain the same number of examples"
        self.text_tokenizer = text_tokenizer
        self.parse_fn = parse_fn
        self.src = src
        self.tgt = tgt
        self.src_sizes = np.array(src_sizes)
        self.tgt_sizes = np.array(tgt_sizes) if tgt_sizes is not None else None
        self.sizes = np.vstack((self.src_sizes, self.tgt_sizes)).T if self.tgt_sizes is not None else self.src_sizes
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.left_pad_source = left_pad_source
        self.left_pad_target = left_pad_target
        self.shuffle = shuffle
        self.input_feeding = input_feeding
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        if num_buckets > 0:
            from fairseq.data import BucketPadLengthDataset
            self.src = BucketPadLengthDataset(
                self.src,
                sizes=self.src_sizes,
                num_buckets=num_buckets,
                pad_idx=self.src_dict.pad(),
                left_pad=self.left_pad_source,
            )
            self.src_sizes = self.src.sizes
            logger.info('bucketing source lengths: {}'.format(list(self.src.buckets)))
            if self.tgt is not None:
                self.tgt = BucketPadLengthDataset(
                    self.tgt,
                    sizes=self.tgt_sizes,
                    num_buckets=num_buckets,
                    pad_idx=self.tgt_dict.pad(),
                    left_pad=self.left_pad_target,
                )
                self.tgt_sizes = self.tgt.sizes
                logger.info('bucketing target lengths: {}'.format(list(self.tgt.buckets)))

            # determine bucket sizes using self.num_tokens, which will return
            # the padded lengths (thanks to BucketPadLengthDataset)
            num_tokens = np.vectorize(self.num_tokens, otypes=[np.long])
            self.bucketed_num_tokens = num_tokens(np.arange(len(self.src)))
            self.buckets = [
                (None, num_tokens)
                for num_tokens in np.unique(self.bucketed_num_tokens)
            ]
        else:
            self.buckets = None
        self.pad_to_multiple = pad_to_multiple

    def get_batch_shapes(self):
        return self.buckets

    def __getitem__(self, index):
        src_str, tgt_str = self.parse_fn(self.src[index])

        example = {
            'id': index,
            'source': src_str,
            'target': tgt_str,
        }
        return example

    def __len__(self):
        return len(self.src)

    def collate(
            self,
            samples,
            pad_idx,
            eos_idx,
            input_feeding=True,
            pad_to_length=None,
            pad_to_multiple=1,
    ):
        if len(samples) == 0:
            return {}

        def merge(key, left_pad, move_eos_to_beginning=False, pad_to_length=None):
            return data_utils.collate_tokens(
                [s[key] for s in samples],
                pad_idx, eos_idx, left_pad, move_eos_to_beginning,
                pad_to_length=pad_to_length,
                pad_to_multiple=pad_to_multiple,
            )

        id = torch.LongTensor([s['id'] for s in samples])

        # We need to manually add bos/eos due to the annoying implementation (https://github.com/huggingface/transformers/issues/7199):
        #    "Please note that the RoBERTa tokenizer is built using only
        #    <s> (the BOS token) and </s> (the SEP token), with two </s></s> as the separator."
        if self.max_source_length is None:
            src_tokens = self.text_tokenizer([s['source'] for s in samples],
                                             add_special_tokens=False) # add <bos> and <eos> later
        else:
            src_tokens = self.text_tokenizer([s['source'] for s in samples],
                                             add_special_tokens=False,
                                             truncation=True,
                                             max_length=self.max_source_length - 2) # account for <bos> and <eos>

        src_lengths = []

        for s, src_token in zip(samples, src_tokens['input_ids']):
            src_token = [self.text_tokenizer.bos_token_id] + src_token + [self.text_tokenizer.eos_token_id]
            s['source'] = torch.LongTensor(src_token)
            src_lengths.append(len(src_token))
        src_tokens = merge(  # [BS x max_len]
            'source', left_pad=self.left_pad_source,
            pad_to_length=pad_to_length['source'] if pad_to_length is not None else None
        )
        # sort by descending source length
        src_lengths = torch.LongTensor(src_lengths)
        src_lengths, sort_order = src_lengths.sort(descending=True)
        id = id.index_select(0, sort_order)
        src_tokens = src_tokens.index_select(0, sort_order)

        prev_output_tokens = None
        target = None
        if samples[0].get('target', None) is not None:
            if self.max_source_length is None:
                src_tokens = self.text_tokenizer([s['source'] for s in samples],
                                                 add_special_tokens=False)
            else:
                tgt_tokens = self.text_tokenizer([s['target'] for s in samples],
                                                 add_special_tokens=False,
                                                 truncation=True,
                                                 max_length=self.max_target_length - 2)
            tgt_lengths = []
            for s, tgt_token in zip(samples, tgt_tokens['input_ids']):
                tgt_token = [self.text_tokenizer.bos_token_id] + tgt_token + [self.text_tokenizer.eos_token_id]
                s['target'] = torch.LongTensor(tgt_token)
                tgt_lengths.append(len(tgt_token))
            target = merge(
                'target', left_pad=self.left_pad_target,
                pad_to_length=pad_to_length['target'] if pad_to_length is not None else None,
            )
            target = target.index_select(0, sort_order)
            tgt_lengths = torch.LongTensor(tgt_lengths).index_select(0, sort_order)
            ntokens = tgt_lengths.sum().item()

            if samples[0].get('prev_output_tokens', None) is not None:
                prev_output_tokens = merge('prev_output_tokens', left_pad=self.left_pad_target)
            elif input_feeding:
                # we create a shifted version of targets for feeding the
                # previous output token(s) into the next decoder step
                prev_output_tokens = merge(
                    'target',
                    left_pad=self.left_pad_target,
                    move_eos_to_beginning=True,
                    pad_to_length=pad_to_length['target'] if pad_to_length is not None else None,
                )
        else:
            ntokens = src_lengths.sum().item()

        batch = {
            'id': id,
            'nsentences': len(samples),
            'ntokens': ntokens,
            'net_input': {
                'src_tokens': src_tokens,
                'src_lengths': src_lengths,
            },
            'target': target,
        }
        if prev_output_tokens is not None:
            batch['net_input']['prev_output_tokens'] = prev_output_tokens.index_select(0, sort_order)

        # remove the 1st useless token
        batch['net_input']['prev_output_tokens'] = batch['net_input']['prev_output_tokens'][:, 1:]
        batch['target'] = batch['target'][:, 1:]
        batch['ntokens'] = batch['ntokens'] - batch['nsentences']

        return batch

    def collater(self, samples, pad_to_length=None):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate
            pad_to_length (dict, optional): a dictionary of
                {'source': source_pad_to_length, 'target': target_pad_to_length}
                to indicate the max length to pad to in source and target respectively.

        Returns:
            dict: a mini-batch with the following keys:

                - `id` (LongTensor): example IDs in the original input order
                - `ntokens` (int): total number of tokens in the batch
                - `net_input` (dict): the input to the Model, containing keys:

                  - `src_tokens` (LongTensor): a padded 2D Tensor of tokens in
                    the source sentence of shape `(bsz, src_len)`. Padding will
                    appear on the left if *left_pad_source* is ``True``.
                  - `src_lengths` (LongTensor): 1D Tensor of the unpadded
                    lengths of each source sentence of shape `(bsz)`
                  - `prev_output_tokens` (LongTensor): a padded 2D Tensor of
                    tokens in the target sentence, shifted right by one
                    position for teacher forcing, of shape `(bsz, tgt_len)`.
                    This key will not be present if *input_feeding* is
                    ``False``.  Padding will appear on the left if
                    *left_pad_target* is ``True``.
        """
        res = self.collate(
            samples,
            pad_idx=self.src_dict.pad(),
            eos_idx=self.src_dict.eos(),
            input_feeding=self.input_feeding,
            pad_to_length=pad_to_length,
            pad_to_multiple=self.pad_to_multiple,
        )

        return res

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return max(self.src_sizes[index], self.tgt_sizes[index] if self.tgt_sizes is not None else 0)

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return (self.src_sizes[index], self.tgt_sizes[index] if self.tgt_sizes is not None else 0)

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            indices = np.random.permutation(len(self)).astype(np.int64)
        else:
            indices = np.arange(len(self), dtype=np.int64)
        if self.buckets is None:
            # sort by target length, then source length
            if self.tgt_sizes is not None:
                indices = indices[
                    np.argsort(self.tgt_sizes[indices], kind='mergesort')
                ]
            return indices[np.argsort(self.src_sizes[indices], kind='mergesort')]
        else:
            # sort by bucketed_num_tokens, which is:
            #   max(padded_src_len, padded_tgt_len)
            return indices[
                np.argsort(self.bucketed_num_tokens[indices], kind='mergesort')
            ]

    @property
    def supports_prefetch(self):
        return (
            getattr(self.src, 'supports_prefetch', False)
            and (getattr(self.tgt, 'supports_prefetch', False) or self.tgt is None)
        )

    def filter_indices_by_size(self, indices, max_sizes):
        """ Filter a list of sample indices. Remove those that are longer
            than specified in max_sizes.

        Args:
            indices (np.array): original array of sample indices
            max_sizes (int or list[int] or tuple[int]): max sample size,
                can be defined separately for src and tgt (then list or tuple)

        Returns:
            np.array: filtered sample array
            list: list of removed indices
        """
        return data_utils.filter_paired_dataset_indices_by_size(
            self.src_sizes,
            self.tgt_sizes,
            indices,
            max_sizes,
        )
