from typing import Union, Optional
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from transformers.trainer_utils import SchedulerType


def get_linear_schedule_with_warmup(
    optimizer: Optimizer,
    num_training_steps: int,
    num_lm_warmup_steps: int,
    num_ke_warmup_steps: int,
    last_epoch=-1
):

    # LM
    def lr_lambda_for_lm(current_step: int):
        if current_step < num_lm_warmup_steps:
            return float(current_step) / float(max(1, num_lm_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_lm_warmup_steps))
        )

    # KE
    def lr_lambda_for_ke(current_step: int):
        if current_step < num_ke_warmup_steps:
            return float(current_step) / float(max(1, num_ke_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_ke_warmup_steps))
        )

    # CLS pooler
    def lr_lambda_for_pooler(current_step: int):
        if current_step < num_lm_warmup_steps:
            return float(current_step) / float(max(1, num_lm_warmup_steps))

        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_lm_warmup_steps))
        )

    return LambdaLR(optimizer, [lr_lambda_for_lm, lr_lambda_for_lm, lr_lambda_for_ke], last_epoch)


TYPE_TO_SCHEDULER_FUNCTION = {
    SchedulerType.LINEAR: get_linear_schedule_with_warmup,
}


def get_scheduler(
    name: str,
    optimizer: Optimizer,
    num_lm_warmup_steps: Optional[int] = None,
    num_ke_warmup_steps: Optional[int] = None,
    num_training_steps: Optional[int] = None,
):

    name = SchedulerType(name)
    schedule_func = TYPE_TO_SCHEDULER_FUNCTION[name]

    # All other schedulers require `num_warmup_steps`
    if num_ke_warmup_steps is None or num_lm_warmup_steps is None:
        raise ValueError(f"{name} requires `num_warmup_steps`, please provide that argument.")

    # All other schedulers require `num_training_steps`
    if num_training_steps is None:
        raise ValueError(f"{name} requires `num_training_steps`, please provide that argument.")

    return schedule_func(
        optimizer,
        num_lm_warmup_steps=num_lm_warmup_steps,
        num_ke_warmup_steps=num_ke_warmup_steps,
        num_training_steps=num_training_steps
    )