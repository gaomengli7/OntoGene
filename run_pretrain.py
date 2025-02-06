import os
import logging
from transformers import HfArgumentParser, set_seed, logging
from transformers import BertTokenizer, AutoTokenizer

from src.models import OntoGenePreTrainedModel
from src.trainer import OntoGeneTrainer
from src.sampling import negative_sampling_strategy
from src.dataset import GeneSeqDataset, GeneGoDataset, GoGoDataset
from src.dataloader import DataCollatorForGoGo, DataCollatorForLanguageModeling, DataCollatorForGeneGo
from src.training_args import OntoGeneModelArguments, OntoGeneDataTrainingArguments, OntoGeneTrainingArguments

logger = logging.get_logger(__name__)
DEVICE = 'cuda'


def main():
    parser = HfArgumentParser((OntoGeneTrainingArguments, OntoGeneDataTrainingArguments, OntoGeneModelArguments))
    training_args, data_args, model_args = parser.parse_args_into_dataclasses()

    os.makedirs(training_args.output_dir, exist_ok=True)

    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        DEVICE,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )

    logger.info(f"Training parameters: {training_args}")

    set_seed(training_args.seed)

    if model_args.gene_model_file_name:
        gene_tokenizer = BertTokenizer.from_pretrained('/OntoGene/data/model_data/DNABERT')
    else:
        raise ValueError("Need provide gene tokenizer config path.")

    text_tokenizer = None
    if model_args.text_model_file_name and training_args.use_desc:
        text_tokenizer = AutoTokenizer.from_pretrained('/OntoGene/data/model_data/PubMedBERT')

    # Load dataset
    gene_seq_dataset = None
    gene_go_dataset = None
    go_go_dataset = None

    negative_sampling_fn = negative_sampling_strategy[data_args.negative_sampling_fn]

    if data_args.model_gene_seq_data:
        gene_seq_dataset = GeneSeqDataset(
            data_dir=data_args.pretrain_data_dir,
            seq_data_path=data_args.gene_seq_data_file_name,
            tokenizer=gene_tokenizer,
            max_gene_seq_length=data_args.max_gene_seq_length,
        )

    if data_args.model_gene_go_data:
        gene_go_dataset = GeneGoDataset(
            data_dir=data_args.pretrain_data_dir,
            use_desc=training_args.use_desc,
            use_seq=training_args.use_seq,
            gene_tokenizer=gene_tokenizer,
            text_tokenizer=text_tokenizer,
            negative_sampling_fn=negative_sampling_fn,
            num_neg_sample=training_args.num_gene_go_neg_sample,
            sample_head=data_args.gene_go_sample_head,
            sample_tail=data_args.gene_go_sample_tail,
            max_gene_seq_length=data_args.max_gene_seq_length,
            max_text_seq_length=data_args.max_text_seq_length
        )

    if data_args.model_go_go_data:
        go_go_dataset = GoGoDataset(
            data_dir=data_args.pretrain_data_dir,
            text_tokenizer=text_tokenizer,
            use_desc=training_args.use_desc,
            negative_sampling_fn=negative_sampling_fn,
            num_neg_sample=training_args.num_go_go_neg_sample,
            sample_head=data_args.go_go_sample_head,
            sample_tail=data_args.go_go_sample_tail,
            max_text_seq_length=data_args.max_text_seq_length,
        )

    # Ontology statistics
    num_relations = gene_go_dataset.num_relations
    num_go_terms = gene_go_dataset.num_go_terms
    num_genes = gene_go_dataset.num_genes


    are_gene_length_same = False
    gene_seq_data_collator = DataCollatorForLanguageModeling(tokenizer=gene_tokenizer, are_gene_length_same=are_gene_length_same)
    gene_go_data_collator = DataCollatorForGeneGo(gene_tokenizer=gene_tokenizer, text_tokenizer=text_tokenizer, are_gene_length_same=are_gene_length_same)
    go_go_data_collator = DataCollatorForGoGo(tokenizer=text_tokenizer)

    model = OntoGenePreTrainedModel.from_pretrained(
        gene_model_path=model_args.gene_model_file_name,
        onto_model_path=model_args.text_model_file_name,
        model_args=model_args,
        training_args=training_args,
        num_relations=num_relations,
        num_go_terms=num_go_terms,
        num_genes=num_genes,
    )


    trainer = OntoGeneTrainer(
        model=model,
        args=training_args,
        gene_seq_dataset=gene_seq_dataset,
        gene_go_dataset=gene_go_dataset,
        go_go_dataset=go_go_dataset,
        gene_seq_data_collator=gene_seq_data_collator,
        gene_go_data_collator=gene_go_data_collator,
        go_go_data_collator=go_go_data_collator
    )

    # Pretraining
    if training_args.do_train:
        trainer.train()


if __name__ == "__main__":
    main()