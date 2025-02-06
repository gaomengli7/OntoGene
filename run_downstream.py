import os
from typing import Optional
from dataclasses import dataclass, field
from transformers import HfArgumentParser, TrainingArguments, BertTokenizerFast, set_seed, Trainer
import logging
from src.benchmark.models import model_mapping, load_sgd_optimizer_and_scheduler
from torch.optim import SGD
from src.benchmark.dataset import dataset_mapping, output_modes_mapping
from src.benchmark.metrics import build_compute_metrics_fn
from src.benchmark.trainer import OntoGeneTrainer
# from torchsummary import summary
# from torchvision.models import DNABERTModel
# myNet=DNABERTModel()
# summary(DNABERTModel,())
import warnings
warnings.filterwarnings("ignore")
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
logger = logging.getLogger(__name__)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# DEVICE = "cuda"
# os.environ['CUDA_LAUNCH_BLOCKING']='1'
@dataclass
class ModelArguments:

    model_name_or_path: str = field(
        default=None,
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )

    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )

    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )

    mean_output: bool = field(
        default=True, metadata={"help": "output of bert, use mean output or pool output"}
    )

    optimizer: str = field(
        default="SGD",
        metadata={"help": "use optimizer: SGD(True) or SGD(False)."}
    )

    frozen_bert: bool = field(
        default=False,
        metadata={"help": "frozen bert model."}
    )
@dataclass
class DynamicTrainingArguments(TrainingArguments):
    # For ensemble
    save_strategy: str = field(
        default='steps',
        metadata={"help": "The checkpoint save strategy to adopt during training."}
    )

    save_steps: int = field(
        default=500,
        metadata={"help": " Number of updates steps before two checkpoint saves"}
    )

    evaluation_strategy: str = field(
        default='steps',
        metadata={"help": "The evaluation strategy to adopt during training."}
    )

    eval_steps: int = field(
        default=100,
        metadata={"help": "Number of update steps between two evaluations"}
    )

    save_logit: bool = field(
        default=False,
        metadata={"help": "Save test file logit with name $TASK-$MODEL_ID-$ARRAY_ID.npy"}
    )

    save_logit_dir: str = field(
        default=None,
        metadata={"help": "Where to save the prediction result"}
    )
    # Regularization
    fix_layers: int = field(
        default=0,
        metadata={"help": "Fix bottom-n layers when optimizing"}
    )

    evaluate_during_training: bool = field(
        default=True,
        metadata={"help": "evaluate during training."}
    )
    save_total_limit: int = field(
        default=3,
        metadata={"help": "If a value is passed, will limit the total amount of checkpoints."}
    )
    # resume_from_checkpoint = True
    fp16 = True
@dataclass
class BTDataTrainingArguments:

    task_name: str = field(metadata={"help": "The name of the task to train on: " + ", ".join(dataset_mapping.keys())})
    data_dir: str = field(
        default=None,
        metadata={"help": "The input data dir. Should contain the .tsv files (or other data files) for the task."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    def __post_init__(self):
        self.task_name = self.task_name.lower()

def main():
    parser = HfArgumentParser((ModelArguments, BTDataTrainingArguments, DynamicTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    test_promotercore_dataset = None
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s",
        training_args.local_rank,
        training_args.n_gpu,
        bool(training_args.local_rank != -1)
    )

    logger.info("Training/evaluation parameters %s", training_args)
    set_seed(training_args.seed)
    try:
        output_mode = output_modes_mapping[data_args.task_name]
        logger.info("Task name: {}, output mode: {}".format(data_args.task_name, output_mode))
    except KeyError:
        raise ValueError("Task not found: %s" % (data_args.task_name))

    # Load dataset
    tokenizer = BertTokenizerFast.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        do_lower_case=False
    )

    processor = dataset_mapping[data_args.task_name](tokenizer=tokenizer)

    num_labels = len(processor.get_labels())
    train_dataset = (
        processor.get_train_examples(data_dir=data_args.data_dir)
    )
    eval_dataset = (
        processor.get_dev_examples(data_dir=data_args.data_dir)
    )
    if data_args.task_name == 'promotercore':
        test_promotercore_dataset = (
            processor.get_test_examples(data_dir=data_args.data_dir, data_cat='test')
        )

    elif data_args.task_name == 'strength':
        test_STRENGTH_dataset = (
            processor.get_test_examples(data_dir=data_args.data_dir, data_cat='test')
        )

    else:
        test_dataset = (
            processor.get_test_examples(data_dir=data_args.data_dir, data_cat='test')
        )

    model_fn = model_mapping[data_args.task_name]  #
    model = model_fn.from_pretrained(
        model_args.model_name_or_path,
        num_labels=num_labels,
        mean_output=model_args.mean_output,

    )
    if model_args.frozen_bert:
        unfreeze_layers = ['layer.29', 'bert.pooler', 'classifier']
        for name, parameters in model.named_parameters():
            parameters.requires_grad = False
            for tags in unfreeze_layers:
                if tags in name:
                    parameters.requires_grad = True
                    break

    if data_args.task_name == 'promotercore' :
        training_args.metric_for_best_model = "eval_accuracy"
    elif data_args.task_name == 'strength':
        training_args.metric_for_best_model = "eval_accuracy"
    else:
        pass
    if data_args.task_name == 'strength':
        # training_args.do_predict=False
        trainer = OntoGeneTrainer(
            # model_init=init_model,
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=build_compute_metrics_fn(data_args.task_name, output_type=output_mode),
            data_collator=train_dataset.collate_fn,
            optimizers=load_sgd_optimizer_and_scheduler(model, training_args,
                                                        train_dataset) if model_args.optimizer == 'SGD' else (
            None, None)
        )
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=build_compute_metrics_fn(data_args.task_name, output_type=output_mode),
            data_collator=train_dataset.collate_fn,
            optimizers=load_sgd_optimizer_and_scheduler(model, training_args,
                                                        train_dataset) if model_args.optimizer == 'SGD' else (
            None, None)
        )
    # Training
    if training_args.do_train:
        # pass
        trainer.train()
        trainer.save_model(training_args.output_dir)
        tokenizer.save_pretrained(training_args.output_dir)
    # Prediction
    logger.info("**** Test ****")
    # trainer.compute_metrics = metrics_mapping(data_args.task_name)

    if test_promotercore_dataset is not None:
        predictions_promotercore, input_ids_promotercore, metrics_promotercore = trainer.predict(
            test_promotercore_dataset)
        print("metrics: ", metrics_promotercore)
    elif data_args.task_name == 'tfbs':
        predictions_STRENGTH, input_ids_STRENGTH, metrics_STRENGTH = trainer.predict(test_STRENGTH_dataset)
        print("metrics: ", metrics_STRENGTH)

    else:
        predictions_family, input_ids_family, metrics_family = trainer.predict(test_dataset)
        print("metrics", metrics_family)

if __name__ == '__main__':
    main()
