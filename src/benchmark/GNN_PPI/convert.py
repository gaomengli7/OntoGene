import os
import json
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import dataclasses
from dataclasses import dataclass
import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel, BertTokenizer
from transformers import PreTrainedTokenizerBase


@dataclass
class GeneSeqInputFeatures:

    input_ids: List[int]
    label: Optional[Union[int, float]] = None

    def to_json_string(self):

        return json.dumps(dataclasses.asdict(self)) + "\n"


def _collate_batch_for_gene_seq(
    examples: List[Dict],
    tokenizer: PreTrainedTokenizerBase,
    are_gene_length_same: bool #
):
    if isinstance(examples[0], GeneSeqInputFeatures):
        examples = [torch.tensor(e.input_ids, dtype=torch.long) for e in examples]

    if are_gene_length_same:
        return torch.stack(examples, dim=0)

    max_length = max(x.size(0) for x in examples)
    result = examples[0].new_full([len(examples), max_length], fill_value=tokenizer.pad_token_id)
    for i, example in enumerate(examples):
        if tokenizer.padding_side == 'right':
            result[i, :example.size(0)] = example
        else:
            result[i, -example.size(0):] = example
    return result


@dataclass
class DataCollatorForLanguageModeling:

    tokenizer: PreTrainedTokenizerBase
    mlm: bool = True
    mlm_probability: float = 0.15
    are_gene_length_same: bool = False

    def __post_init__(self):
        if self.mlm and self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. "
                "You should pass `mlm=False` to train on causal language modeling instead."
            )
    
    def __call__(
        self,
        examples: List[Dict],
    ) -> Dict[str, torch.Tensor]:
        batch = {'input_ids': _collate_batch_for_gene_seq(examples, self.tokenizer, self.are_gene_length_same)}
        special_tokens_mask = batch.pop('special_tokens_mask', None)
        if self.mlm:
            batch['input_ids'], batch['labels'] = self.mask_tokens(
                batch['input_ids'], special_tokens_mask=special_tokens_mask
            )
        else:
            labels = batch['input_ids'].clone()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            batch['labels'] = labels

        batch['attention_mask'] = (batch['input_ids'] != self.tokenizer.pad_token_id).long()
        batch['token_type_ids'] = torch.zeros_like(batch['input_ids'], dtype=torch.long)
        return batch

    def mask_tokens(
        self,
        inputs: torch.Tensor,
        special_tokens_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        labels = inputs.clone()
        probability_matrix = torch.full(labels.size(), fill_value=self.mlm_probability)
        # if `special_tokens_mask` is None, generate it by `labels`
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        # only compute loss on masked tokens.
        labels[~masked_indices] = -100

        # 80% of the time, replace masked input tokens with tokenizer.mask_token
        indices_replaced = torch.bernoulli(torch.full(labels.shape, fill_value=0.8)).bool() & masked_indices #
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, fill_value=0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        return inputs, labels


class GeneSeqDataset(Dataset):

    def __init__(
        self,
        file_path: str,
        seq_data_path: str = None,
        tokenizer: PreTrainedTokenizerBase = None,
        in_memory: bool=True,
        max_gene_seq_length: int = None
    ):
        self.file_path = file_path
        self.seq_data_path = seq_data_path

        self.gene_seq = [line.rstrip('\n').split('\t')[1] for line in open(file_path, 'r')]

        self.tokenizer = tokenizer
        self.max_gene_seq_length = max_gene_seq_length
        
    def __getitem__(self, index):
        item = self.gene_seq[index]

        # implement padding of sequences at 'DataCollatorForLanguageModeling'
        item = list(item)
        if self.max_gene_seq_length is not None:
            item = item[:self.max_gene_seq_length]
        input_ids = self.tokenizer.encode(item)
        return GeneSeqInputFeatures(
            input_ids=input_ids,
        )
        
    def __len__(self):
        # return self.num_examples
        return len(self.gene_seq)


class Seq2Vec(nn.Module):
    def __init__(
        self,
        pretrained_model_path: str,
    ):
        super().__init__()

        onto_gene_model = BertModel.from_pretrained(pretrained_model_path)
        self.encoder = onto_gene_model

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        outputs = self.encoder( #
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=return_dict,
        )
        gene_attention_mask = attention_mask.bool()
        num_batch_size = attention_mask.size(0)
        gene_embedding = torch.stack([outputs.last_hidden_state[i, gene_attention_mask[i, :], :][1:-1].mean(dim=0) for i in range(num_batch_size)], dim=0)
        return gene_embedding

def convert_gene_seq_to_embedding(
    file_path: str,
    pretrained_model_path: str,
    embedding_save_path: str
):

    tokenizer = BertTokenizer.from_pretrained(pretrained_model_path)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, are_gene_length_same=False)

    # Note: default set gene length to 1024.
    gene_seq_dataset = GeneSeqDataset(
        file_path=file_path,
        tokenizer=tokenizer,
        max_gene_seq_length=1024
    )

    gene_seq_dataloader = DataLoader(
        dataset=gene_seq_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        collate_fn=data_collator
    )

    model = Seq2Vec(pretrained_model_path=pretrained_model_path)
    model.to('cuda:6')

    def to_device(inputs: Dict[str, torch.Tensor]):
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to('cuda:6')
        return inputs
    
    gene_embeddings = []
    for item in tqdm.tqdm(gene_seq_dataloader):
        _ = item.pop('labels')
        inputs = to_device(item)
        with torch.no_grad():
            gene_embedding = model(**inputs).cpu()
            gene_embeddings.append(gene_embedding)
    
    gene_embeddings = torch.cat(gene_embeddings, dim=0)
    
    np.save(embedding_save_path, gene_embeddings)


if __name__ == '__main__':
    convert_gene_seq_to_embedding(
        'data/gene.sequences.dictionary.tsv',
        'data/datasets/model/ontogene',
        'data/gene_embedding_ontogene.npy'
    )