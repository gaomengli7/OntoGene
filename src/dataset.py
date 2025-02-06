import os
from posixpath import join
from sys import path
import time
import lmdb
import torch
import json
import numpy as np
import pickle as pkl
import dataclasses
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


def _split_go_by_type(go_types) -> Dict[str, List]:
    TAS_go = []
    HDA_go = []
    IDA_go = []
    IEA_go = []
    ISS_go = []
    IMP_go = []
    IBA_go = []
    IPI_go = []
    NAS_go = []
    HMP_go = []
    IC_go = []
    IGA_go = []
    ISA_go = []
    ISO_go = []
    EXP_go = []
    RCA_go = []
    ND_go = []
    IKR_go = []
    ISM_go = []
    IGI_go = []
    IEP_go = []
    for go_id, type_ in go_types.items():
        if type_ == 'TAS':
            TAS_go.append(go_id)
        elif type_ == 'HDA':
            HDA_go.append(go_id)
        elif type_ == 'IDA':
            IDA_go.append(go_id)
        elif type_ == 'IEA':
            IEA_go.append(go_id)
        elif type_ == 'ISS':
            ISS_go.append(go_id)
        elif type_ == 'IMP':
            IMP_go.append(go_id)
        elif type_ == 'IBA':
            IBA_go.append(go_id)
        elif type_ == 'IPI':
            IPI_go.append(go_id)
        elif type_ == 'NAS':
            NAS_go.append(go_id)
        elif type_ == 'HMP':
            HMP_go.append(go_id)
        elif type_ == 'IC':
            IC_go.append(go_id)
        elif type_ == 'IGA':
            IGA_go.append(go_id)
        elif type_ == 'ISA':
            ISA_go.append(go_id)
        elif type_ == 'ISO':
            ISO_go.append(go_id)
        elif type_ == 'EXP':
            EXP_go.append(go_id)
        elif type_ == 'RCA':
            RCA_go.append(go_id)
        elif type_ == 'IKR':
            IKR_go.append(go_id)
        elif type_ == 'ISM':
            ISM_go.append(go_id)
        elif type_ == 'IGI':
            IGI_go.append(go_id)
        elif type_ == 'IEP':
            IEP_go.append(go_id)
        elif type_ == 'ND':
            ND_go.append(go_id)
        else:
            # print(go_id, type_, '错误的')
            raise Exception('the type not supported.')

    go_terms_type_dict = {
        'TAS': TAS_go,
        'HDA': HDA_go,
        'IDA': IDA_go,
        'IEA': IEA_go,
        'ISS': ISS_go,
        'IMP': IMP_go,
        'IBA': IBA_go,
        'IPI': IPI_go,
        'NAS': NAS_go,
        'HMP': HMP_go,
        'IC': IC_go,
        'IGA': IGA_go,
        'ISA': ISA_go,
        'ISO': ISO_go,
        'EXP': EXP_go,
        'RCA': RCA_go,
        'ND': ND_go,
        'IKR': IKR_go,
        'ISM': ISM_go,
        'IGI': IGI_go,
        'IEP': IEP_go,
    }

    return go_terms_type_dict


def get_triplet_data(data_path):
    heads = []
    relations = []
    tails = []
    true_tail = {}
    true_head = {}

    for line in open(data_path, 'r'):
        head, relation, tail = [int(id) for id in line.rstrip('\n').split()]
        heads.append(head)
        relations.append(relation)
        tails.append(tail)

        if (head, relation) not in true_tail:
            true_tail[(head, relation)] = []
        true_tail[(head, relation)].append(tail)
        if (relation, tail) not in true_head:
            true_head[(relation, tail)] = []
        true_head[(relation, tail)].append(head)

    true_tail = {key: np.array(list(set(val))) for key, val in true_tail.items()}
    true_head = {key: np.array(list(set(val))) for key, val in true_head.items()}
    return heads, relations, tails, true_tail, true_head


@dataclass
class GeneGoInputFeatures:
    """
    A single set of feature of data for OntoGene pretrain.
    """
    postive_gene_input_ids: List[int]
    postive_relation_ids: int
    postive_go_input_ids: Union[int, List[int]]
    negative_gene_input_ids: List[List[int]] = None
    negative_gene_attention_mask: Optional[List[int]] = None
    negative_relation_ids: List[int] = None
    negative_go_input_ids: List[Union[int, List[int]]] = None

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self)) + "\n"


@dataclass
class GoGoInputFeatures:
    """
    A single set of feature of data for Go-GO triplet in OntoGene pretrain.
    """
    postive_go_head_input_ids: Union[int, List[int]]
    postive_relation_ids: int
    postive_go_tail_input_ids: Union[int, List[int]]
    negative_go_head_input_ids: List[Union[int, List[int]]] = None
    negative_relation_ids: List[int] = None
    negative_go_tail_input_ids: List[Union[int, List[int]]] = None

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self)) + "\n"


@dataclass
class GeneSeqInputFeatures:
    """
    A single set of feature of data for gene sequences.
    """
    input_ids: List[int]
    label: Optional[Union[int, float]] = None


    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self)) + "\n"


class GeneGoDataset(Dataset):


    def __init__(
            self,
            data_dir: str,
            use_seq: bool,
            use_desc: bool,
            gene_tokenizer: PreTrainedTokenizerBase = None,
            text_tokenizer: PreTrainedTokenizerBase = None,
            negative_sampling_fn=None,
            num_neg_sample: int = 1,
            sample_head: bool = False,
            sample_tail: bool = True,
            max_gene_seq_length: int = None,
            max_text_seq_length: int = None
    ):
        self.data_dir = data_dir
        self.use_seq = use_seq
        self.use_desc = use_desc
        self._load_data()

        self.gene_tokenizer = gene_tokenizer
        self.text_tokenizer = text_tokenizer
        self.negative_sampling_fn = negative_sampling_fn
        self.num_neg_sample = num_neg_sample
        self.sample_head = sample_head
        self.sample_tail = sample_tail
        self.max_gene_seq_length = max_gene_seq_length
        self.max_text_seq_length = max_text_seq_length

    def _load_data(self):


        self.go2id = [line.rstrip('\n') for line in open(os.path.join(self.data_dir, 'go2id.txt'), 'r')]
        self.relation2id = [line.rstrip('\n') for line in open(os.path.join(self.data_dir, 'relation2id.txt'), 'r')]

        self.num_go_terms = len(self.go2id)
        self.num_relations = len(self.relation2id)

        self.go_types = {idx: line.rstrip('\n') for idx, line in
                         enumerate(open(os.path.join(self.data_dir, 'go_evidence1.txt'), 'r'))}
        # print(self.go_types,"*"*100)
        # exit()
        self.gene_seq = [line.rstrip('\n') for line in open(os.path.join(self.data_dir, 'gene_seq_new.txt'), 'r')]
        self.num_genes = len(self.gene_seq)

        if self.use_desc:
            self.go_descs = {idx: line.rstrip('\n') for idx, line in
                             enumerate(open(os.path.join(self.data_dir, 'go_def.txt'), 'r'))}

        # split go term according to ontology type.
        self.go_terms_type_dict = _split_go_by_type(self.go_types)

        # for negative sample.
        self.gene_heads, self.pg_relations, self.go_tails, self.true_tail, self.true_head = get_triplet_data(
            data_path=os.path.join(self.data_dir, 'gene_go_triplet.txt')
        )

    def __getitem__(self, index):
        gene_head_id, relation_id, go_tail_id = self.gene_heads[index], self.pg_relations[index], self.go_tails[
            index]

        gene_input_ids = gene_head_id

        # use sequence.
        if self.use_seq:

            # tokenize gene sequence.
            gene_head_seq = list(self.gene_seq[gene_head_id])

            if self.max_gene_seq_length is not None:
                gene_head_seq = gene_head_seq[:self.max_gene_seq_length]
            gene_input_ids = self.gene_tokenizer.encode(list(gene_head_seq))

        go_tail_type = self.go_types[go_tail_id]

        go_input_ids = go_tail_id
        if self.use_desc:
            go_desc = self.go_descs[go_tail_id]
            go_input_ids = self.text_tokenizer.encode(go_desc, max_length=self.max_text_seq_length, truncation=True,
                                                      padding='max_length')

        negative_gene_input_ids_list = []
        negative_relation_ids_list = []
        negative_go_input_ids_list = []

        if self.sample_tail:
            tail_negative_samples = self.negative_sampling_fn(
                cur_entity=(gene_head_id, relation_id),
                num_neg_sample=self.num_neg_sample,
                true_triplet=self.true_tail,
                num_entity=None,
                go_terms=self.go_terms_type_dict[go_tail_type]
            )

            for neg_go_id in tail_negative_samples:
                neg_go_input_ids = neg_go_id
                if self.use_desc:
                    neg_go_desc = self.go_descs[neg_go_id]
                    neg_go_input_ids = self.text_tokenizer.encode(neg_go_desc, max_length=self.max_text_seq_length,
                                                                  truncation=True, padding='max_length')

                negative_gene_input_ids_list.append(gene_input_ids)
                negative_relation_ids_list.append(relation_id)
                negative_go_input_ids_list.append(neg_go_input_ids)

        if self.sample_head:
            head_negative_samples = self.negative_sampling_fn(
                cur_entity=(relation_id, go_tail_id),
                num_neg_sample=self.num_neg_sample,
                true_triplet=self.true_head,
                num_entity=self.num_genes,
                go_terms=None,
            )

            for neg_gene_id in head_negative_samples:
                neg_gene_input_ids = neg_gene_id
                if self.use_seq:
                    neg_gene_seq = list(self.gene_heads[neg_gene_id])
                    if self.max_gene_seq_length is not None:
                        neg_gene_seq = neg_gene_seq[:self.max_gene_seq_length]
                    neg_gene_input_ids = self.gene_tokenizer.encode(neg_gene_seq)

                negative_gene_input_ids_list.append(neg_gene_input_ids)
                negative_relation_ids_list.append(relation_id)
                negative_go_input_ids_list.append(go_input_ids)

        assert len(negative_gene_input_ids_list) == len(negative_relation_ids_list)
        assert len(negative_relation_ids_list) == len(negative_go_input_ids_list)

        return GeneGoInputFeatures(
            postive_gene_input_ids=gene_input_ids,
            postive_relation_ids=relation_id,
            postive_go_input_ids=go_input_ids,
            negative_gene_input_ids=negative_gene_input_ids_list,
            negative_relation_ids=negative_relation_ids_list,
            negative_go_input_ids=negative_go_input_ids_list
        )

    def __len__(self):
        assert len(self.gene_heads) == len(self.pg_relations)
        assert len(self.pg_relations) == len(self.go_tails)

        return len(self.gene_heads)

    def get_num_go_terms(self):
        return len(self.go_types)

    def get_num_gene_go_relations(self):
        return len(list(set(self.pg_relations)))


class GoGoDataset(Dataset):

    def __init__(
            self,
            data_dir: str,
            use_desc: bool = False,
            text_tokenizer: PreTrainedTokenizerBase = None,
            negative_sampling_fn=None,
            num_neg_sample: int = 1,
            sample_head: bool = True,
            sample_tail: bool = True,
            max_text_seq_length: int = None
    ):
        self.data_dir = data_dir
        self.use_desc = use_desc
        self.text_tokenizer = text_tokenizer
        self.negative_sampling_fn = negative_sampling_fn
        self.num_neg_sample = num_neg_sample
        self.sample_head = sample_head
        self.sample_tail = sample_tail
        self.max_text_seq_length = max_text_seq_length
        self._load_data()

    def _load_data(self):

        self.go2id = [line.rstrip('\n') for line in open(os.path.join(self.data_dir, 'go2id.txt'), 'r')]
        self.relation2id = [line.rstrip('\n') for line in open(os.path.join(self.data_dir, 'relation2id.txt'), 'r')]
        self.num_go_terms = len(self.go2id)
        self.num_relations = len(self.relation2id)  #21
        self.go_types = {idx: line.rstrip('\n') for idx, line in
                         enumerate(open(os.path.join(self.data_dir, 'go_evidence1.txt'), 'r'))}
        if self.use_desc:
            self.go_descs = {idx: line.rstrip('\n') for idx, line in
                             enumerate(open(os.path.join(self.data_dir, 'go_def.txt'), 'r'))}


        self.go_terms_type_dict = _split_go_by_type(self.go_types)
        self.go_heads, self.gg_relations, self.go_tails, self.true_tail, self.true_head = get_triplet_data(
            data_path=os.path.join(self.data_dir, 'go_go_triplet.txt')
        )

    def __getitem__(self, index):
        go_head_id, relation_id, go_tail_id = self.go_heads[index], self.gg_relations[index], self.go_tails[index]

        go_head_type = self.go_types[go_head_id]
        go_tail_type = self.go_types[go_tail_id]
        go_head_input_ids = go_head_id
        go_tail_input_ids = go_tail_id
        if self.use_desc:
            go_head_desc = self.go_descs[go_head_id]
            go_tail_desc = self.go_descs[go_tail_id]
            go_head_input_ids = self.text_tokenizer.encode(go_head_desc, padding='max_length', truncation=True,
                                                           max_length=self.max_text_seq_length)
            go_tail_input_ids = self.text_tokenizer.encode(go_tail_desc, padding='max_length', truncation=True,
                                                           max_length=self.max_text_seq_length)

        negative_go_head_input_ids_list = []
        negative_relation_ids_list = []
        negative_go_tail_input_ids_list = []

        if self.sample_tail:
            tail_negative_samples = self.negative_sampling_fn(
                cur_entity=(go_head_id, relation_id),
                num_neg_sample=self.num_neg_sample,
                true_triplet=self.true_tail,
                num_entity=None,
                go_terms=self.go_terms_type_dict[go_tail_type]
            )

            for neg_go_id in tail_negative_samples:
                neg_go_input_ids = neg_go_id
                if self.use_desc:
                    neg_go_desc = self.go_descs[neg_go_id]
                    neg_go_input_ids = self.text_tokenizer.encode(neg_go_desc, max_length=self.max_text_seq_length,
                                                                  truncation=True, padding='max_length')

                negative_go_head_input_ids_list.append(go_head_input_ids)
                negative_relation_ids_list.append(relation_id)
                negative_go_tail_input_ids_list.append(neg_go_input_ids)

        if self.sample_head:
            head_negative_samples = self.negative_sampling_fn(
                cur_entity=(relation_id, go_tail_id),
                num_neg_sample=self.num_neg_sample,
                true_triplet=self.true_head,
                num_entity=None,
                go_terms=self.go_terms_type_dict[go_head_type]
            )

            for neg_go_id in head_negative_samples:
                neg_go_input_ids = neg_go_id
                if self.use_desc:
                    neg_go_desc = self.go_descs[neg_go_id]
                    neg_go_input_ids = self.text_tokenizer.encode(neg_go_desc, max_length=self.max_text_seq_length,
                                                                  truncation=True, padding='max_length')

                negative_go_head_input_ids_list.append(neg_go_input_ids)
                negative_relation_ids_list.append(relation_id)
                negative_go_tail_input_ids_list.append(go_tail_input_ids)

        assert len(negative_go_head_input_ids_list) == len(negative_relation_ids_list)
        assert len(negative_relation_ids_list) == len(negative_go_tail_input_ids_list)

        return GoGoInputFeatures(
            postive_go_head_input_ids=go_head_input_ids,
            postive_relation_ids=relation_id,
            postive_go_tail_input_ids=go_tail_input_ids,
            negative_go_head_input_ids=negative_go_head_input_ids_list,
            negative_relation_ids=negative_relation_ids_list,
            negative_go_tail_input_ids=negative_go_tail_input_ids_list
        )

    def __len__(self):
        assert len(self.go_heads) == len(self.gg_relations)
        assert len(self.gg_relations) == len(self.go_tails)

        return len(self.go_heads)

    def get_num_go_terms(self):
        return len(self.go_types)

    def get_num_go_go_relations(self):
        return len(list(set(self.gg_relations)))


class GeneSeqDataset(Dataset):


    def __init__(
            self,
            data_dir: str,
            seq_data_path: str = None,
            tokenizer: PreTrainedTokenizerBase = None,
            in_memory: bool = True,
            max_gene_seq_length: int = None
    ):
        self.data_dir = data_dir
        self.seq_data_path = seq_data_path

        self.gene_seq = [line.rstrip('\n') for line in open(os.path.join(self.data_dir, 'gene_seq_new.txt'), 'r')]
        self.tokenizer = tokenizer
        self.max_gene_seq_length = max_gene_seq_length

    def __getitem__(self, index):

        item = self.gene_seq[index]

        input_ids = self.tokenizer.encode(item)
        first_number = input_ids[0]
        last_number = input_ids[-1]
        input_ids = [first_number] + input_ids[1:-1][:self.max_gene_seq_length - 2] + [last_number]



        return GeneSeqInputFeatures(
            input_ids=input_ids,
        )

    def __len__(self):
        # return self.num_examples
        return len(self.gene_seq)

