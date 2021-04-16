# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging
import os

from dataclasses import dataclass, field
from fairseq.data.encoders import register_bpe
from fairseq.dataclass import FairseqDataclass

logger = logging.getLogger(__name__)

@dataclass
class HuggingFacePretrainedBPEConfig(FairseqDataclass):
    bpe_vocab: str = field(default="???", metadata={"help": "path to vocab.json"})
    bpe_merges: str = field(default="???", metadata={"help": "path to merges.txt"})
    bpe_dropout: float = field(default=0.0, metadata={"help": "BPE dropout rate"})

    # parser.add_argument('--pretrained-vocab', required=True, help='vocab of pretrained model name')
    # parser.add_argument('--cache-dir', default=None, help='path to save vocab cache')
    # parser.add_argument('--special-vocab', default=None, help='path to special vocab')
    # parser.add_argument('--bpe-dropout', default=0.0, type=float, help='Rate for BPE dropout')

@register_bpe('hf_pretrained_bpe', dataclass=HuggingFacePretrainedBPEConfig)
class HuggingFacePretrainedBPE(object):
    @staticmethod
    def load(args):
        try:
            # from transformers import AutoTokenizer
            from transformers import RobertaTokenizer, RobertaTokenizerFast, AddedToken
        except ImportError:
            raise ImportError(
                'Please install huggingface/tokenizers with: '
                'pip install transformers'
            )

        logger.info('Loading pretrained vocabulary from %s' % args.bpe_vocab)
        # tokenizer = RobertaTokenizerFast(vocab_file=args.bpe_vocab, merges_file=args.bpe_merges)

        # initialize a slow tokenizer and convert it to fast, so that special tokens can be properly segmented
        # hard-code <sep>
        sep_token = '<sep>'
        kp_special_tokens = ['<present>', '<absent>',
                             '<category>', '<infill>', '<seealso>', '<header>',
                             '<|endoftext|>', '<sep>', '<mask>',
                             '<mixed>', '<number>', '<phrase>']

        tokenizer = RobertaTokenizer(vocab_file=args.bpe_vocab,
                                     merges_file=args.bpe_merges,
                                     sep=sep_token,  # doesn't matter
                                     additional_special_tokens=kp_special_tokens)
        sep_token_id = tokenizer.convert_tokens_to_ids(sep_token)
        added_sep_token = AddedToken(sep_token, lstrip=False, rstrip=False)
        tokenizer.sep_token = sep_token
        tokenizer._sep_token = added_sep_token
        tokenizer.init_kwargs['sep_token'] = sep_token
        tokenizer.all_special_ids.append(sep_token_id)
        tokenizer.all_special_tokens.append(sep_token)
        tokenizer.all_special_tokens_extended.append(added_sep_token)
        tokenizer.special_tokens_map['sep_token'] = sep_token
        tokenizer.special_tokens_map_extended['sep_token'] = added_sep_token

        tokenizer.unique_no_split_tokens = tokenizer.all_special_tokens

        tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base",
                                                         __slow_tokenizer=tokenizer, tokenizer_file=None,
                                                         vocab_file=args.bpe_vocab,
                                                         merges_file=args.bpe_merges)
        logger.info('Vocab size=%d, base vocab size=%d' % (len(tokenizer), tokenizer.vocab_size))

        # a workaround for bpe dropout (https://github.com/huggingface/tokenizers/issues/201)
        if float(args.bpe_dropout) > 0.0:
            workaround_files = tokenizer._tokenizer.model.save(os.path.dirname(args.bpe_vocab), 'workaround')
            tokenizer._tokenizer.model = type(tokenizer._tokenizer.model)(*workaround_files, dropout=float(args.bpe_dropout))

        return tokenizer

    def encode(self, x: str, *inputs, **kwargs) -> list:
        return self.tokenizer.encode(x, *inputs, **kwargs)

    def decode(self, x: list, *inputs, **kwargs) -> str:
        return self.tokenizer.decode(x, *inputs, **kwargs)

    def is_beginning_of_word(self, x: str) -> bool:
        # TODO not right for RoBERTa BPE
        return not x.startswith('##')
