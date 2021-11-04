# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import json
from functools import lru_cache

import numpy as np

from fairseq.tasks.keyphrasification_utils import KP_DATASET_FIELDS
from . import FairseqDataset
from fairseq.file_io import PathManager


class RawTextDataset(FairseqDataset):
    """Modified on the basis of data.indexed_dataset.IndexedRawTextDataset"""

    def __init__(self, filepath, text_field=None):
        """

        :param filepath:
        :param text_field: If not given, take each line as a data point.
                           If given, presumably it's a jsonl file and take the data of given field as data point.
        """
        self.filepath = filepath
        self.text_field = text_field

        self.lines = None
        self.ex_sizes = None # length of each example
        self._size = None # number of examples

    @staticmethod
    def read_data(filepath, text_field):
        lines = []
        if text_field is None:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f: lines.append(line.strip())
        else:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f: lines.append(json.loads(line)[text_field])

        # print('Loaded %d data from: %s' % (len(lines), filepath))
        return lines

    def check_index(self, i):
        if i < 0 or i >= len(self):
            raise IndexError('index out of range')

    @lru_cache(maxsize=64)
    def __getitem__(self, i):
        self.check_index(i)
        return self.all_examples[i]

    def get_original_text(self, i):
        self.check_index(i)
        return self.example_dicts[i]

    def __del__(self):
        pass

    def __len__(self):
        if not self._size:
            self._size = len(self.all_examples)

        return self._size

    def num_tokens(self, index):
        return self.sizes[index]

    def size(self, index):
        return self.sizes[index]

    @property
    def sizes(self):
        if self.ex_sizes is None:
            self.ex_sizes = np.asarray([len(l.split()) * 1.5 for l in self.all_examples])
        return self.ex_sizes

    @property
    def all_examples(self):
        # lazy load
        if not self.lines:
            self.lines = self.read_data(self.filepath, self.text_field)
        return self.lines

    @staticmethod
    def exists(path):
        return PathManager.exists(path)

    @property
    def supports_prefetch(self):
        return False