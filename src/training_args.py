from collections import defaultdict
import math
from dataclasses import dataclass, field
from transformers import logging
from transformers.training_args import TrainingArguments

from src.sampling import negative_sampling_strategy


@dataclass
class OntoGeneModelArguments:
    gene_model_file_name: str = field(
        default=None,
        metadata={"help": "The directory of gene sequence pretrained model."}
    )
    text_model_file_name: str = field(
        default=None,
        metadata={"help": "The directory of text sequence pretrained model."}
    )
    gene_model_config_name: str = field(
        default=None,
        metadata={'help': "Gene pretrained config name or path if not the same as gene_model_file_name"}
    )
    text_model_config_name: str = field(
        default=None,
        metadata={"help": "Text pretrained config name or path if not the same as text_model_file_name"}
    )
    gene_tokenizer_name: str = field(
        default=None,
        metadata={"help": "Gene sequence tokenizer name or path if not the same as gene_model_file_name"}
    )
    text_tokenizer_name: str = field(
        default=None,
        metadata={"help": "Text sequence tokenizer name or path if not the same as text_model_file_name"}
    )

    # For OntoModel
    go_encoder_cls: str = field(
        default='embedding',
        metadata={"help": "The class of Go term description encoder"}
    )
    gene_encoder_cls: str = field(
        default='bert',
        metadata={'help': 'The class of gene encoder.'}
    )
    ke_embedding_size: int = field(
        default=1024,
        metadata={"help": "Size of knowledge embedding when using `Embedding` as Go encoder."}
    )
    double_entity_embedding_size: bool = field(
        default=False,
        metadata={"help": "Whether or not to set the entity embedding size to double."}
    )


@dataclass
class OntoGeneTrainingArguments(TrainingArguments):

    optimize_memory: bool = field(
        default=False,
        metadata={"help": "Whether or not to optimize memory when computering the loss function of negative samples. "}
    )

    use_seq: bool = field(
        default=True,
        metadata={"help": "Whether or not to use gene sequence, which its pooler output through encoder as gene representation."}
    )

    use_desc: bool = field(
        default=False,
        metadata={"help": "Whether or not to use description of Go term, which its pooler output through encoder as Go term embedding."}
    )

    dataloader_gene_go_num_workers: int = field(
        default=1,
        metadata={"help": "Number of workers to collate gene-go dataset."}
    )
    dataloader_go_go_num_workers: int = field(
        default=1,
        metadata={"help": "Number of workers to collate go-go dataset."}
    )
    dataloader_gene_seq_num_workers: int = field(
        default=1,
        metadata={'help': "Number of workers to collate gene sequence dataset."}
    )

    # number of negative sampling
    num_gene_go_neg_sample: int = field(
        default=1,
        metadata={"help": "Number of negatve sampling for Gene-Go"}
    )
    num_go_go_neg_sample: int = field(
        default=1,
        metadata={"help": "Number of negative sampling for Go-Go"}
    )

    # Weight of KE loss and MLM loss in total loss
    ke_lambda: float = field(
        default=1.0,
        metadata={"help": "Weight of KE loss."}
    )
    mlm_lambda: float = field(
        default=1.0,
        metadata={"help": "Weight of KE loss."}
    )

    # margin in KE score function.
    ke_score_fn: str = field(
        default=1.0,
        metadata={"help": "Type of score function."}
    )
    ke_max_score: float = field(
        default=1.0,
        metadata={"help": "Margin in KE score function."}
    )

    # respectively set learning rate to training of gene language model and knowledge embedding
    lm_learning_rate: float = field(
        default=5e-5,
        metadata={"help": "The initial MLM learning rate for AdamW."}
    )
    ke_learning_rate: float = field(
        default=1e-4,
        metadata={"help": "the initial KE learning rate for AdamW."}
    )

    num_gene_seq_epochs: int = field(
        default=3,
        metadata={"help": "Total number of training epochs of Gene MLM to perform."}
    )
    num_gene_go_epochs: int = field(
        default=3,
        metadata={"help": "Total number of training epochs of Gene-Go KE to perform."}
    )
    num_go_go_epochs: int = field(
        default=3,
        metadata={"help": "Total number of training epochs of Go-Go KE to perform."}
    )

    per_device_train_gene_seq_batch_size: int = field(
        default=8,
        metadata={"help": "Batch size per GPU/TPU core/CPU for training of Gene MLM."}
    )
    per_device_train_gene_go_batch_size: int = field(
        default=8,
        metadata={"help": "Batch size per GPU/TPU core/CPU for training of Gene-Go KE."}
    )
    per_device_train_go_go_batch_size: int = field(
        default=8,
        metadata={"help": "Batch size per GPU/TPU core/CPU for training of Go-Go KE."}
    )

    max_steps: int = field(
        default=-1,
        metadata={"help": "If > 0: set total number of training steps to perform. Override num_train_epochs."}
    )

    # distinguish steps of linear warmup on LM and KE.
    lm_warmup_steps: int = field(
        default=0,
        metadata={"help": "Linear warmup over warmup_steps for LM."}
    )
    ke_warmup_steps: int = field(
        default=0,
        metadata={"help": "Linear warmup over warmup_steps for KE."}
    )
    lm_warmup_ratio: float = field(
        default=0.0,
        metadata={"help": "Linear warmup over warmup_ratio fraction of total steps for LM."}
    )
    ke_warmup_ratio: float = field(
        default=0.0,
        metadata={"help": "Linear warmup over warmup_ratio fraction of total steps for KE."}
    )

    def __post_init__(self):
        super().__post_init__()

        self.per_device_train_gene_seq_batch_size = self.per_device_train_batch_size
        self.per_device_train_go_go_batch_size = self.per_device_train_batch_size
        self.per_device_train_gene_go_batch_size = self.per_device_train_batch_size


    @property
    def train_gene_seq_batch_size(self) -> int:

        per_device_batch_size = self.per_device_train_gene_seq_batch_size
        train_batch_size = per_device_batch_size * max(1, self.n_gpu)
        return train_batch_size

    @property
    def train_gene_go_batch_size(self) -> int:

        per_device_batch_size = self.per_device_train_gene_go_batch_size
        train_batch_size = per_device_batch_size * max(1, self.n_gpu)
        return train_batch_size

    @property
    def train_go_go_batch_size(self) -> int:

        per_device_batch_size = self.per_device_train_go_go_batch_size
        train_batch_size = per_device_batch_size * max(1, self.n_gpu)
        return train_batch_size

    def get_warmup_steps(self, num_training_steps: int):

        warmup_steps = (
            self.warmup_steps if self.warmup_steps > 0 else math.ceil(num_training_steps * self.warmup_ratio)
        )
        return warmup_steps

    def get_lm_warmup_steps(self, num_training_steps: int):

        warmup_steps = (
            self.lm_warmup_steps if self.lm_warmup_steps > 0 else math.ceil(num_training_steps * self.lm_warmup_ratio)
        )
        return warmup_steps

    def get_ke_warmup_steps(self, num_training_steps: int):

        warmup_steps = (
            self.ke_warmup_steps if self.lm_warmup_steps > 0 else math.ceil(num_training_steps * self.ke_warmup_ratio)
        )
        return warmup_steps


@dataclass
class OntoGeneDataTrainingArguments:

    model_gene_seq_data: bool = field(
        default=True,
        metadata={"help": "Whether or not to model gene sequence data."}
    )
    model_gene_go_data: bool = field(
        default=True,
        metadata={"help": "Whether or not to model triplet data of `Gene-Go`"}
    )
    model_go_go_data: bool = field(
        default=True,
        metadata={"help": "Whether or not to model triplet data of `Go-Go`"}
    )


    pretrain_data_dir: str = field(
        default='/OntoGene/data/GeneKG',
        metadata={"help": "the directory path of pretrain data."}
    )
    gene_seq_data_file_name: str = field(
        default='gene_seq',
        metadata={"help": "the directory path of specific gene sequence data."}
    )
    in_memory: bool = field(
        default=False,
        metadata={"help": "Whether or not to save data into memory during sampling"}
    )

    # negative sampling
    negative_sampling_fn: str = field(
        default="simple_random",
        metadata={"help": f"Strategy of negative sampling. Could choose {', '.join(negative_sampling_strategy.keys())}"}
    )
    gene_go_sample_head: bool = field(
        default=False,
        metadata={"help": "Whether or not to sample head entity in triplet of `gene-go`"}
    )
    gene_go_sample_tail: bool = field(
        default=True,
        metadata={"help": "Whether or not to sample tail entity in triplet of `gene-go`"}
    )
    go_go_sample_head: bool = field(
        default=False,
        metadata={"help": "Whether or not to sample head entity in triplet of `go-go`"}
    )
    go_go_sample_tail: bool = field(
        default=False,
        metadata={"help": "Whether or not to sample tail entity in triplet of `go-go`"}
    )


    max_gene_seq_length: int = field(
        default=None,
        metadata={"help": "Maximum length of gene sequence."}
    )
    max_text_seq_length: int = field(
        default=512,
        metadata={"help": "Maximum length of Go term description."}
    )


