# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

import numpy as np
import torch
from transformers import PreTrainedTokenizer

from fairseq.data import Dictionary, data_utils, FairseqDataset


logger = logging.getLogger(__name__)


def get_word_spans(tokens, delimiter):
    word_start_positions = [tid for tid, t in enumerate(tokens) if t.startswith(delimiter)]

    word_spans = []

    for i, word_pos in enumerate(word_start_positions):
        if i == 0: # first word
            if word_start_positions[0] > 0:
                word_spans.append((0, word_start_positions[0]))
            word_spans.append((word_start_positions[i], word_start_positions[i + 1]))
        elif i == len(word_start_positions) - 1: # last word
            if word_start_positions[-1] <= len(tokens):
                word_spans.append((word_start_positions[-1], len(tokens)))
        else:
            word_spans.append((word_start_positions[i], word_start_positions[i + 1]))

    return word_spans


class MlmOtfDataset(FairseqDataset):
    """
    A pair of torch.utils.data.Datasets.

    Args:
        dataset (torch.utils.data.Dataset): source dataset to wrap
        data_sizes (List[int]): source sentence lengths
        vocab (~fairseq.data.Dictionary): source vocabulary
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
        self,
        dataset: torch.utils.data.Dataset,
        data_sizes: list,
        vocab: Dictionary,
        text_tokenizer: PreTrainedTokenizer,
        shuffle: bool,
        epoch: int,
        seed: int = 1,
        tokens_per_sample: int = 512,
        mask_prob: float = 0.15,
        leave_unmasked_prob: float = 0.1,
        random_token_prob: float = 0.1,
        split: str = 'valid',
        no_bos_eos: bool = False
    ):
        assert 0.0 < mask_prob < 1.0
        assert 0.0 <= random_token_prob <= 1.0
        assert 0.0 <= leave_unmasked_prob <= 1.0
        assert random_token_prob + leave_unmasked_prob <= 1.0

        self.dataset = dataset
        self.data_sizes = data_sizes

        self.vocab = vocab
        self.text_tokenizer = text_tokenizer

        self.shuffle = shuffle
        self.seed = seed
        self.epoch = epoch

        self.tokens_per_sample = tokens_per_sample
        self.mask_prob = mask_prob
        self.leave_unmasked_prob = leave_unmasked_prob
        self.random_token_prob = random_token_prob

        self.buckets = None
        self.split = split
        self.no_bos_eos = no_bos_eos

    def get_batch_shapes(self):
        return self.buckets

    def __getitem__(self, index):
        example = {
            'id': index,
            'text': self.dataset[index],
        }
        return example

    def __len__(self):
        return len(self.dataset)

    def collate(
            self,
            samples,
            pad_idx,
            eos_idx,
            pad_to_length=None,
    ):
        if len(samples) == 0:
            return {}

        def merge(key, left_pad, move_eos_to_beginning=False, pad_to_length=None):
            return data_utils.collate_tokens(
                [s[key] for s in samples],
                pad_idx, eos_idx, left_pad, move_eos_to_beginning,
                pad_to_length=pad_to_length
            )

        id = torch.LongTensor([s['id'] for s in samples])

        # We need to manually add bos/eos due to the annoying implementation (https://github.com/huggingface/transformers/issues/7199):
        #    "Please note that the RoBERTa tokenizer is built using only
        #    <s> (the BOS token) and </s> (the SEP token), with two </s></s> as the separator."
        if self.tokens_per_sample is None:
            tokenized_samples = self.text_tokenizer([s['text'] for s in samples],
                                             add_special_tokens=False) # add <bos> and <eos> later
        else:
            tokenized_samples = self.text_tokenizer([s['text'] for s in samples],
                                             add_special_tokens=False,
                                             truncation=True,
                                             max_length=self.tokens_per_sample - 2) # account for <bos> and <eos>

        lengths = []

        cur_seed = (self.seed + self.epoch) if self.shuffle else 0

        with data_utils.numpy_seed(cur_seed):
            for sample, tokened_sample in zip(samples, tokenized_samples.encodings):
                # TODO: whole-word masking currently only works for RoBERTa tokenizer
                tokens = np.asarray(tokened_sample.tokens)
                word_spans = get_word_spans(tokens, delimiter='Ġ')

                num_token = len(tokens)
                num_word = len(word_spans)

                token_mask = np.full(num_token, False)
                # target side, unmasked positions are [pad]
                tgt_ids = np.full(num_token, self.text_tokenizer.pad_token_id).tolist()
                src_ids = tokened_sample.ids
                src_ids_nomask = tokened_sample.ids

                # sample word masks
                num_masked_word = int(self.mask_prob * num_word + np.random.rand())
                masked_word_idx = sorted(np.random.choice(num_word, num_masked_word, replace=False))
                rand_or_unmask_prob = self.random_token_prob + self.leave_unmasked_prob

                # source: replace masked part with [mask]/random tokens/unchanged
                for masked_word_id in masked_word_idx:
                    b, e = word_spans[masked_word_id]
                    token_mask[b: e] = True
                    tgt_ids[b: e] = tokened_sample.ids[b: e]
                    roll = np.random.rand()
                    if roll < rand_or_unmask_prob:
                        if roll < self.random_token_prob:
                            # replace with random tokens
                            src_ids[b: e] = np.random.choice(
                                len(self.vocab),
                                size=(e - b)
                            )
                        else:
                            # leave as is
                            continue
                    else:
                        # replace with [mask]
                        src_ids[b: e] = [self.text_tokenizer.mask_token_id] * (e - b)

                if not self.no_bos_eos:
                    src_ids = [self.text_tokenizer.bos_token_id] + src_ids + [self.text_tokenizer.eos_token_id]
                    tgt_ids = [self.text_tokenizer.pad_token_id] + tgt_ids + [self.text_tokenizer.pad_token_id]

                sample['source'] = torch.LongTensor(src_ids)
                sample['source_nomask'] = torch.LongTensor(src_ids_nomask)
                sample['target'] = torch.LongTensor(tgt_ids)
                assert sample['source'].shape == sample['target'].shape, 'size of source/target must match!'
                lengths.append(len(src_ids))

            # if self.split == 'valid':
            #     print(self.text_tokenizer.decode(src_ids))
            #     print(self.text_tokenizer.decode(tgt_ids))

            src_ids = merge(  # [BS x max_len]
                'source', left_pad=False, pad_to_length=pad_to_length['source'] if pad_to_length is not None else None)
            tgt_ids = merge(  # [BS x max_len]
                'target', left_pad=False, pad_to_length=pad_to_length['target'] if pad_to_length is not None else None)
            src_ids_nomask = merge(  # [BS x max_len]
                'source_nomask', left_pad=False, pad_to_length=pad_to_length['source'] if pad_to_length is not None else None)


        # sort by descending source length
        lengths = torch.LongTensor(lengths)
        lengths, sort_order = lengths.sort(descending=True)
        id = id.index_select(0, sort_order)
        src_ids = src_ids.index_select(0, sort_order)
        tgt_ids = tgt_ids.index_select(0, sort_order)
        src_ids_nomask = src_ids_nomask.index_select(0, sort_order)

        prev_output_tokens = None
        ntokens = lengths.sum().item()

        batch = {
            'id': id,
            'nsentences': len(samples),
            'ntokens': ntokens,
            'net_input': {
                'src_tokens': src_ids,
                'src_lengths': lengths,
                'src_tokens_nomask': src_ids_nomask,
            },
            'target': tgt_ids,
        }
        if prev_output_tokens is not None:
            batch['net_input']['prev_output_tokens'] = prev_output_tokens.index_select(0, sort_order)

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
            pad_to_length=pad_to_length,
        )

        return res

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching.

        Since sequences are truncated by Huggingface Tokenizer, we cap the length by tokens_per_sample.
        """
        return min(self.data_sizes[index], self.tokens_per_sample)

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return (self.src_sizes[index], self.tgt_sizes[index] if self.tgt_sizes is not None else 0)

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            with data_utils.numpy_seed(self.seed + self.epoch):
                indices = np.random.permutation(len(self)).astype(np.int64)
        else:
            indices = np.arange(len(self), dtype=np.int64)

        return indices[np.argsort(self.data_sizes[indices], kind='mergesort')]

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

