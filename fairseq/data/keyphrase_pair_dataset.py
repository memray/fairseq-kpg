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
        dataset (torch.utils.data.Dataset): source dataset to wrap
        sizes (List[int]): source sentence lengths
        vocab (~fairseq.data.Dictionary): source vocabulary
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
        self, dataset, vocab, sizes,
        text_tokenizer, transform_fns,
        input_feeding=True,
        left_pad_source=False, left_pad_target=False,
        max_source_length=None, max_target_length=None,
        shuffle=True, sort_by_length=True,
        pad_to_multiple=1
    ):
        self.text_tokenizer = text_tokenizer
        self.transform_fns = transform_fns
        self.dataset = dataset
        self.sizes = np.array(sizes)
        self.vocab = vocab
        self.left_pad_source = left_pad_source
        self.left_pad_target = left_pad_target
        self.input_feeding = input_feeding
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.buckets = None
        self.shuffle = shuffle
        self.sort_by_length = sort_by_length
        self.pad_to_multiple = pad_to_multiple

    def get_batch_shapes(self):
        return self.buckets

    def __getitem__(self, index):
        example = self.dataset[index]
        example['id'] = index

        for transform_fn in self.transform_fns:
            example = transform_fn(example)

        return example

    def __len__(self):
        return len(self.dataset)

    def collate(
            self,
            samples,
            pad_idx,
            eos_idx,
            input_feeding=True,
            pad_to_length=None,
            pad_to_multiple=1,
    ):
        num_sample = len(samples)
        samples = [s for s in samples if len(s['source'].strip()) > 0 and len(s['target'].strip()) > 0] # filter empty-target examples
        # print('#sample: pre-filter=%d, after-filter=%d, diff=%d' % (num_sample, len(samples), num_sample-len(samples)))

        if len(samples) == 0:
            return {}
        # print('=*' * 50)
        # for s in samples:
        #     print('id', s['id'])
        #     print('source', len(s['source']), s['source'])
        #     print('target', len(s['target']), s['target'])
        #     print('=' * 50)

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
        try:
            if self.max_source_length is None:
                src_tokens = self.text_tokenizer([s['source'] for s in samples],
                                                 add_special_tokens=False) # add <bos> and <eos> later
            else:
                src_tokens = self.text_tokenizer([s['source'] for s in samples],
                                                 add_special_tokens=False,
                                                 truncation=True,
                                                 max_length=self.max_source_length - 2) # account for <bos> and <eos>
        except Exception:
            # occasionally
            return {}

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
            try:
                if self.max_target_length is None:
                    tgt_tokens = self.text_tokenizer([s['target'] for s in samples],
                                                     add_special_tokens=False)
                else:
                    tgt_tokens = self.text_tokenizer([s['target'] for s in samples],
                                                     add_special_tokens=False,
                                                     truncation=True,
                                                     max_length=self.max_target_length - 2)
            except Exception:
                return {}

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

        # print(src_lengths.numpy().mean(), tgt_lengths.numpy().mean())

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
            pad_idx=self.vocab.pad(),
            eos_idx=self.vocab.eos(),
            input_feeding=self.input_feeding,
            pad_to_length=pad_to_length,
            pad_to_multiple=self.pad_to_multiple,
        )

        return res

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching.

        Since sequences are truncated by Huggingface Tokenizer, we cap the length by max_source_length/max_target_length.
        """
        return min(self.sizes[index], self.max_source_length)

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return self.sizes[index]

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            indices = np.random.permutation(len(self)).astype(np.int64)
        else:
            indices = np.arange(len(self), dtype=np.int64)
        if self.sort_by_length:
            # sort by target length, then source length
            return indices[np.argsort(self.sizes[indices], kind='mergesort')]
        if self.buckets is not None:
            # sort by bucketed_num_tokens, which is:
            #   max(padded_src_len, padded_tgt_len)
            return indices[
                np.argsort(self.bucketed_num_tokens[indices], kind='mergesort')
            ]
        return indices

    @property
    def supports_prefetch(self):
        return (
            getattr(self.src, 'supports_prefetch', False)
            and (getattr(self.tgt, 'supports_prefetch', False) or self.tgt is None)
        )

    def filter_indices_by_size(self, indices, max_sizes):
        """ Functionally disabled. Was used to filter a list of sample indices, removing those that are longer than specified in max_sizes.
            Since long sequences are truncated by Huggingface Tokenizer, we disable its function here.

        Args:
            indices (np.array): original array of sample indices
            max_sizes (int or list[int] or tuple[int]): max sample size,
                can be defined separately for src and tgt (then list or tuple)

        Returns:
            np.array: filtered sample array
            list: list of removed indices
        """
        return indices, []
