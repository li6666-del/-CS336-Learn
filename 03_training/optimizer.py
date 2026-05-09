import math
from dataclasses import dataclass
from typing import Iterable

import torch


@dataclass
class AdamWConfig:
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8


@dataclass
class CosineLRScheduleConfig:
    max_lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 1000
    max_steps: int = 100000


def get_cosine_lr(step: int, config: CosineLRScheduleConfig) -> float:
    if step < 0:
        raise ValueError(f"step must be non-negative, got {step}")

    if config.warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")

    if config.max_steps <= 0:
        raise ValueError("max_steps must be positive")

    if config.min_lr > config.max_lr:
        raise ValueError("min_lr must be less than or equal to max_lr")

    if config.warmup_steps > 0 and step < config.warmup_steps:
        return config.max_lr * (step + 1) / config.warmup_steps

    if step >= config.max_steps:
        return config.min_lr

    decay_steps = config.max_steps - config.warmup_steps
    if decay_steps <= 0:
        return config.min_lr

    decay_step = step - config.warmup_steps
    decay_ratio = decay_step / decay_steps
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))

    return config.min_lr + coeff * (config.max_lr - config.min_lr)


def set_learning_rate(optimizer: torch.optim.Optimizer, learning_rate: float) -> None:
    for param_group in optimizer.param_groups:
        param_group["lr"] = learning_rate


def configure_adamw(
    parameters: Iterable[torch.nn.Parameter],
    config: AdamWConfig,
) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        parameters,
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        eps=config.eps,
        weight_decay=config.weight_decay,
    )


def configure_adamw_for_model(
    model: torch.nn.Module,
    config: AdamWConfig,
) -> torch.optim.AdamW:
    decay_params = []
    no_decay_params = []

    for param in model.parameters():
        if not param.requires_grad:
            continue

        if param.ndim >= 2:
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    param_groups = [
        {
            "params": decay_params,
            "weight_decay": config.weight_decay,
        },
        {
            "params": no_decay_params,
            "weight_decay": 0.0,
        },
    ]

    return torch.optim.AdamW(
        param_groups,
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        eps=config.eps,
    )
