from pathlib import Path
# from plistlib import Dict
from typing import Union
import pickle as pkl
import lmdb
import numpy as np
import pandas as pd
import re
import torch
from scipy.spatial.distance import squareform, pdist
from tape.datasets import pad_sequences, dataset_factory
from torch.utils.data import Dataset
import os
from torch.utils.data import Subset

class LMDBDataset(Dataset):
    def __init__(self, data_file, in_memory):
        env = lmdb.open(data_file, max_readers=1, readonly=True,
                        lock=False, readahead=False, meminit=False)
        with env.begin(write=False) as txn:
            num_examples = pkl.loads(txn.get(b'num_examples'))
        if in_memory:
            cache = [None] * num_examples
            self._cache = cache
        self._env = env
        self._in_memory = in_memory
        self._num_examples = num_examples

    def __len__(self):
        return self._num_examples

    def __getitem__(self, index):
        if self._in_memory and self._cache[index] is not None:
            item = self._cache[index]
        else:
            with self._env.begin(write=False) as txn:
                item = pkl.loads(txn.get(str(index).encode()))
                if 'id' not in item:
                    item['id'] = str(index)
                if self._in_memory:
                    self._cache[index] = item
        return item


class DataProcessor:

    def get_train_examples(self, data_dir):
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        raise NotImplementedError()

    def get_labels(self):
        raise NotImplementedError()


class PromotercoreProgress(DataProcessor):
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer

    def get_train_examples(self, data_dir, in_memory=True):
        dataset = PromotercoreDataset(data_dir, split='train', tokenizer=self.tokenizer)
        return dataset

    def get_dev_examples(self, data_dir, in_memory=True):
        dataset = PromotercoreDataset(data_dir, split='valid', tokenizer=self.tokenizer)
        return dataset

    def get_test_examples(self, data_dir, data_cat, in_memory=True):
        if data_cat is not None:
            dataset = PromotercoreDataset(data_dir, split=data_cat, tokenizer=self.tokenizer)
        else:
            dataset = PromotercoreDataset(data_dir, split='test', tokenizer=self.tokenizer)
        return dataset

    def get_labels(self):
        return list(range(2))


class PromotercoreDataset(Dataset):
    def __init__(self, file_path, split, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.file_path = file_path  # datasets
        self.max_length = max_length
        if split not in ('train', 'valid', 'test'):
            raise ValueError(f"Unrecognized split: {split}. Must be one of "
                             f"['train', 'valid', 'test']")

        data_file = f'{self.file_path}/promotercore/promotercore_{split}.json'
        self.seqs, self.labels = self.get_data(data_file)

    def get_data(self, file):
        fp = pd.read_json(file)
        seqs = fp.seqs
        labels = fp.label
        return seqs, labels
        # print(labels,"*"*100)
        # exit()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        seq_str = self.seqs[index]
        seq_list = seq_str.split()
        input_ids = self.tokenizer(seq_list, is_split_into_words=True, truncation=True, padding="max_length",
                                   max_length=512
                                   )
        input_ids = np.array(
            input_ids['input_ids'])

        input_mask = np.ones_like(input_ids)
        label = self.labels[index]
        return {'input_ids': input_ids, 'input_mask': input_mask, 'label': label}

    def collate_fn(self, batch):
        input_ids_batch = torch.tensor([item['input_ids'] for item in batch])
        attention_mask_batch = torch.tensor([item['input_mask'] for item in batch])

        fold_labels = [item['label'] for item in batch]
        fold_labels = [int(label) if isinstance(label, (int, str)) and str(label).isdigit() else 0 for label in
                       fold_labels]
        fold_label_batch = torch.LongTensor(fold_labels)
        original_labels = [item.get('original_labels', None) for item in batch]
        return {'input_ids': input_ids_batch,
                'attention_mask': attention_mask_batch,
                'labels': fold_label_batch,
                }


output_modes_mapping = {
    'promotercore': 'sequence-level-classification',

}

dataset_mapping = {
    'promotercore': PromotercoreProgress,

}
