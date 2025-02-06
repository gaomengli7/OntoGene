


import importlib.util
import io
import json
import sys
import weakref
from copy import deepcopy
from functools import partialmethod
sys.path.append('/home/gaomengli/.conda/envs/ontoprotein2/lib/python3.8/site-packages/transformers')
sys.path.append('/home/gaomengli/.conda/envs/ontoprotein2/lib/python3.8/site-packages/transformers')
sys.path.append('/home/gaomengli/.conda/envs/ontoprotein2/lib/python3.8/site-packages/transformers/utils')
import dependency_versions_check
from dependency_versions_check  import dep_version_check
import file_utils
from  file_utils import is_torch_available
import utils
from utils import logging
#from .dependency_versions_check import dep_version_check
#from .file_utils import is_torch_available
#from .utils import logging


if is_torch_available():
    import torch

logger = logging.get_logger(__name__)


def is_deepspeed_available():
    return importlib.util.find_spec("deepspeed") is not None


class HfDeepSpeedConfig:


    def __init__(self, config_file_or_dict):
        # set global weakref object
        set_hf_deepspeed_config(self)

        dep_version_check("deepspeed")

        if isinstance(config_file_or_dict, dict):

            config = deepcopy(config_file_or_dict)
        elif isinstance(config_file_or_dict, str):
            with io.open(config_file_or_dict, "r", encoding="utf-8") as f:
                config = json.load(f)
        else:
            raise ValueError("expecting either a path to a DeepSpeed config file or a pre-populated dict")
        self.config = config


        self._stage = self.get_value("zero_optimization.stage", -1)


        self._offload = False
        if self.is_zero2() or self.is_zero3():
            offload_devices_valid = set(["cpu", "nvme"])
            offload_devices = set(
                [
                    self.get_value("zero_optimization.offload_optimizer.device"),
                    self.get_value("zero_optimization.offload_param.device"),
                ]
            )
            if len(offload_devices & offload_devices_valid) > 0:
                self._offload = True

    def find_config_node(self, ds_key_long):
        config = self.config

        nodes = ds_key_long.split(".")
        ds_key = nodes.pop()
        for node in nodes:
            config = config.get(node)
            if config is None:
                return None, ds_key

        return config, ds_key

    def get_value(self, ds_key_long, default=None):
        config, ds_key = self.find_config_node(ds_key_long)
        if config is None:
            return default
        return config.get(ds_key, default)

    def is_true(self, ds_key_long):
        value = self.get_value(ds_key_long)
        return False if value is None else bool(value)

    def is_false(self, ds_key_long):

        value = self.get_value(ds_key_long)
        return False if value is None else not bool(value)

    def is_zero2(self):
        return self._stage == 2

    def is_zero3(self):
        return self._stage == 3

    def is_offload(self):
        return self._offload


class HfTrainerDeepSpeedConfig(HfDeepSpeedConfig):


    def __init__(self, config_file_or_dict):
        super().__init__(config_file_or_dict)
        self._dtype = torch.float16
        self.mismatches = []

    def dtype(self):
        return self._dtype

    def fill_match(self, ds_key_long, hf_val, hf_key=None, must_match=True):

        config, ds_key = self.find_config_node(ds_key_long)
        if config is None:
            return

        if config.get(ds_key) == "auto":
            config[ds_key] = hf_val
            return

        if not must_match:
            return

        ds_val = config.get(ds_key)
        if ds_val is not None and ds_val != hf_val:
            self.mismatches.append(f"- ds {ds_key_long}={ds_val} vs hf {hf_key}={hf_val}")

    fill_only = partialmethod(fill_match, must_match=False)

    def trainer_config_process(self, args):

        train_batch_size = args.world_size * args.per_device_train_batch_size * args.gradient_accumulation_steps
        self.fill_match(
            "train_micro_batch_size_per_gpu", args.per_device_train_batch_size, "per_device_train_batch_size"
        )
        self.fill_match("gradient_accumulation_steps", args.gradient_accumulation_steps, "gradient_accumulation_steps")
        self.fill_match("train_batch_size", train_batch_size, "train_batch_size (calculated)")
        self.fill_match("gradient_clipping", args.max_grad_norm, "max_grad_norm")

        self.fill_match("optimizer.params.lr", args.learning_rate, "learning_rate")
        self.fill_match("optimizer.params.betas", [args.adam_beta1, args.adam_beta2], "adam_beta1+adam_beta2")
        self.fill_match("optimizer.params.eps", args.adam_epsilon, "adam_epsilon")
        self.fill_match("optimizer.params.weight_decay", args.weight_decay, "weight_decay")

        self.fill_only("scheduler.params.warmup_min_lr", 0)
        self.fill_match("scheduler.params.warmup_max_lr", args.learning_rate, "learning_rate")

        if args.fp16:
            fp16_backend = "apex" if args.fp16_backend == "apex" else "amp"
        else:
            fp16_backend = None


        self.fill_match("fp16.enabled", fp16_backend == "amp", "fp16+fp16_backend(amp)")
        self.fill_match("amp.enabled", fp16_backend == "apex", "fp16+fp16_backend(apex)")
        self.fill_match("amp.opt_level", args.fp16_opt_level, "fp16_opt_level")
        if self.is_false("fp16.enabled"):
            self._dtype = torch.float32

    def trainer_config_finalize(self, args, model, num_training_steps):

        if self.is_zero3():
            hidden_size = model.config.hidden_size
            self.fill_only("zero_optimization.reduce_bucket_size", hidden_size * hidden_size)
            self.fill_only("zero_optimization.stage3_prefetch_bucket_size", 0.9 * hidden_size * hidden_size)
            self.fill_only("zero_optimization.stage3_param_persistence_threshold", 10 * hidden_size)

        # scheduler
        self.fill_match("scheduler.params.total_num_steps", num_training_steps, "num_training_steps (calculated)")
        self.fill_match("scheduler.params.warmup_num_steps", args.get_warmup_steps(num_training_steps), "warmup_steps")

        if len(self.mismatches) > 0:
            mismatches = "\n".join(self.mismatches)
            raise ValueError(
                f"Please correct the following DeepSpeed config values that mismatch TrainingArguments values:\n{mismatches}\n"
                "The easiest method is to set these DeepSpeed config values to 'auto'."
            )


_hf_deepspeed_config_weak_ref = None


def set_hf_deepspeed_config(hf_deepspeed_config_obj):
    global _hf_deepspeed_config_weak_ref

    _hf_deepspeed_config_weak_ref = weakref.ref(hf_deepspeed_config_obj)


def is_deepspeed_zero3_enabled():
    if _hf_deepspeed_config_weak_ref is not None and _hf_deepspeed_config_weak_ref() is not None:
        return _hf_deepspeed_config_weak_ref().is_zero3()
    else:
        return False


def deepspeed_config():
    if _hf_deepspeed_config_weak_ref is not None and _hf_deepspeed_config_weak_ref() is not None:
        return _hf_deepspeed_config_weak_ref().config
    else:
        return None


def deepspeed_init(trainer, num_training_steps, resume_from_checkpoint=None):
    import deepspeed
    from deepspeed.utils import logger as ds_logger

    model = trainer.model
    args = trainer.args

    hf_deepspeed_config = args.hf_deepspeed_config
    hf_deepspeed_config.trainer_config_finalize(args, model, num_training_steps)

    config = hf_deepspeed_config.config


    optimizer = None
    if "optimizer" in config:
        if args.adafactor:
            raise ValueError(
                "--adafactor was passed, but also found `optimizer` configured in the DeepSpeed config. "
                "Only one optimizer can be configured."
            )
        
    else:
        if hf_deepspeed_config.is_offload():
            logger.info(
                "Detected ZeRO Offload and non-DeepSpeed optimizers: This combination should work as long as the custom optimizer has both CPU and GPU implementation (except LAMB)"
            )

        trainer.create_optimizer()
        optimizer = trainer.optimizer


        config["zero_allow_untested_optimizer"] = True


    lr_scheduler = None
    if "scheduler" not in config:
        if "optimizer" in config:

            raise ValueError("At the moment HF scheduler + DeepSpeed optimizer combination is not possible")
        else:
            trainer.create_scheduler(num_training_steps=num_training_steps)
            lr_scheduler = trainer.lr_scheduler


    ds_logger.setLevel(args.get_process_log_level())

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())

    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        model_parameters=model_parameters,
        config_params=config,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
    )

    if resume_from_checkpoint is not None:
        import glob

        deepspeed_checkpoint_dirs = sorted(glob.glob(f"{resume_from_checkpoint}/global_step*"))

        if len(deepspeed_checkpoint_dirs) > 0:
            logger.info(f"Attempting to resume from {resume_from_checkpoint}")

            load_path, _ = model.load_checkpoint(
                resume_from_checkpoint, load_optimizer_states=True, load_lr_scheduler_states=True
            )
            if load_path is None:
                raise ValueError(f"[deepspeed] failed to resume from checkpoint {resume_from_checkpoint}")
        else:
            logger.info(f"{resume_from_checkpoint} doesn't have deepspeed checkpoints, doing nothing")

    return model, optimizer, lr_scheduler
