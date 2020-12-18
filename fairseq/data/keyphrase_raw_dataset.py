# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import json
from functools import lru_cache
import os
import shutil
import struct

import numpy as np
import torch

from fairseq.tasks import keyphrasification
from . import FairseqDataset
from fairseq.data.fasta_dataset import FastaDataset
from fairseq.file_io import PathManager


class KeyphraseRawDataset(FairseqDataset):
    """Modified on the basis of data.indexed_dataset.IndexedRawTextDataset
    Takes a text file as input and binarizes it in memory at instantiation.
    Original lines are not kept in memory"""

    def __init__(self, filepath, dataset_type):
        self.filepath = filepath
        if not dataset_type:
            dataset_type = self.infer_dataset_type(filepath)
        self.dataset_type = dataset_type

        self.example_dicts = None
        self._sizes = None
        self._size = None


    def infer_dataset_type(self, filepath):
        dataset_type = None
        if 'stack' in filepath:
            dataset_type = 'qa'
        elif 'openkp' in filepath:
            dataset_type = 'webpage'
        elif 'times' in filepath:
            dataset_type = 'news'
        elif 'kp20k' in filepath or 'magkp' in filepath:
            dataset_type = 'scipaper'

        assert dataset_type is not None, 'Fail to detect the data type of the given input file.' \
                                         'Accecpted values:' + keyphrasification.KP_DATASET_FIELDS.keys()

        print('Automatically detect the input data type as ' + dataset_type.upper())

        return dataset_type

    def read_data(self, filepath):
        print('Loading data from ' + filepath)
        ex_dicts = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f: ex_dicts.append(json.loads(line))

        return ex_dicts

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
        if not self._sizes:
            title_field, text_field, _, _ = keyphrasification.KP_DATASET_FIELDS[self.dataset_type]
            self._sizes = np.asarray([len(l[title_field].split() + l[text_field].split()) for l in self.all_examples])

        return self._sizes

    @property
    def all_examples(self):
        # lazy load
        if not self.example_dicts:
            self.example_dicts = self.read_data(self.filepath)
        return self.example_dicts

    @staticmethod
    def exists(path):
        return PathManager.exists(path)
