import argparse
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "01_bpe"))
sys.path.insert(0, str(ROOT / "02_model"))
sys.path.insert(0, str(ROOT / "03_training"))

from bpe_tokenizer import Tokenizer
from data import get_batch
from losses import cross_entropy_loss
from optimizer import (
    AdamWConfig,
    CosineLRScheduleConfig,
    configure_adamw_for_model,
    get_cosine_lr,
    set_learning_rate,
)
from transformer_made import TransformerConfig, TransformerLM


@dataclass
class TrainConfig:
    data_path: Path = ROOT / "data" / "TinyStoriesV2-GPT4-valid.txt"
    train_data_path: Path | None = None
    valid_data_path: Path | None = None
    vocab_path: Path = ROOT / "01_bpe" / "vocab.bin"
    merges_path: Path = ROOT / "01_bpe" / "merges.bin"
    token_cache_path: Path = ROOT / "data" / "tinystories_tokens.pt"
    train_token_cache_path: Path | None = None
    valid_token_cache_path: Path | None = None
    checkpoint_dir: Path = ROOT / "checkpoints"
    rebuild_token_cache: bool = False

    vocab_size: int = 0
    context_length: int = 256
    d_model: int = 384
    num_layers: int = 6
    num_heads: int = 6
    d_ff: int = 1536
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-5

    batch_size: int = 16
    max_steps: int = 5000
    eval_interval: int = 250
    eval_iters: int = 20
    checkpoint_interval: int = 1000
    train_split: float = 0.9
    grad_clip: float = 1.0
    seed: int = 1337

    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 200
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8

    device: str = "auto"


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train a small decoder-only Transformer LM.")

    parser.add_argument("--data_path", type=Path, default=TrainConfig.data_path)
    parser.add_argument("--train_data_path", type=Path, default=TrainConfig.train_data_path)
    parser.add_argument("--valid_data_path", type=Path, default=TrainConfig.valid_data_path)
    parser.add_argument("--vocab_path", type=Path, default=TrainConfig.vocab_path)
    parser.add_argument("--merges_path", type=Path, default=TrainConfig.merges_path)
    parser.add_argument("--token_cache_path", type=Path, default=TrainConfig.token_cache_path)
    parser.add_argument("--train_token_cache_path", type=Path, default=TrainConfig.train_token_cache_path)
    parser.add_argument("--valid_token_cache_path", type=Path, default=TrainConfig.valid_token_cache_path)
    parser.add_argument("--checkpoint_dir", type=Path, default=TrainConfig.checkpoint_dir)
    parser.add_argument("--rebuild_token_cache", action="store_true")

    parser.add_argument("--vocab_size", type=int, default=TrainConfig.vocab_size)
    parser.add_argument("--context_length", type=int, default=TrainConfig.context_length)
    parser.add_argument("--d_model", type=int, default=TrainConfig.d_model)
    parser.add_argument("--num_layers", type=int, default=TrainConfig.num_layers)
    parser.add_argument("--num_heads", type=int, default=TrainConfig.num_heads)
    parser.add_argument("--d_ff", type=int, default=TrainConfig.d_ff)
    parser.add_argument("--rope_theta", type=float, default=TrainConfig.rope_theta)
    parser.add_argument("--rms_norm_eps", type=float, default=TrainConfig.rms_norm_eps)

    parser.add_argument("--batch_size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--max_steps", type=int, default=TrainConfig.max_steps)
    parser.add_argument("--eval_interval", type=int, default=TrainConfig.eval_interval)
    parser.add_argument("--eval_iters", type=int, default=TrainConfig.eval_iters)
    parser.add_argument("--checkpoint_interval", type=int, default=TrainConfig.checkpoint_interval)
    parser.add_argument("--train_split", type=float, default=TrainConfig.train_split)
    parser.add_argument("--grad_clip", type=float, default=TrainConfig.grad_clip)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)

    parser.add_argument("--learning_rate", type=float, default=TrainConfig.learning_rate)
    parser.add_argument("--min_lr", type=float, default=TrainConfig.min_lr)
    parser.add_argument("--warmup_steps", type=int, default=TrainConfig.warmup_steps)
    parser.add_argument("--weight_decay", type=float, default=TrainConfig.weight_decay)
    parser.add_argument("--beta1", type=float, default=TrainConfig.beta1)
    parser.add_argument("--beta2", type=float, default=TrainConfig.beta2)
    parser.add_argument("--eps", type=float, default=TrainConfig.eps)

    parser.add_argument("--device", type=str, default=TrainConfig.device)

    return TrainConfig(**vars(parser.parse_args()))


def resolve_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def config_to_dict(config: TrainConfig) -> dict:
    result = asdict(config)
    for key, value in result.items():
        if isinstance(value, Path):
            result[key] = str(value)
    return result


def load_token_data(
    data_path: Path,
    token_cache_path: Path,
    tokenizer: Tokenizer,
    rebuild_token_cache: bool,
) -> torch.Tensor:
    if token_cache_path.exists() and not rebuild_token_cache:
        tokens = torch.load(token_cache_path, map_location="cpu")
        return tokens.long()

    text = data_path.read_text(encoding="utf-8")
    token_ids = tokenizer.encode(text)
    tokens = torch.tensor(token_ids, dtype=torch.long)

    token_cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tokens, token_cache_path)

    return tokens


def split_data(data: torch.Tensor, train_split: float) -> tuple[torch.Tensor, torch.Tensor]:
    if not 0.0 < train_split < 1.0:
        raise ValueError(f"train_split must be between 0 and 1, got {train_split}")

    split_idx = int(len(data) * train_split)
    return data[:split_idx], data[split_idx:]


def load_train_valid_data(
    config: TrainConfig,
    tokenizer: Tokenizer,
) -> tuple[torch.Tensor, torch.Tensor]:
    has_separate_paths = (
        config.train_data_path is not None
        or config.valid_data_path is not None
    )

    if has_separate_paths:
        if config.train_data_path is None or config.valid_data_path is None:
            raise ValueError(
                "train_data_path and valid_data_path must be provided together."
            )

        train_cache_path = (
            config.train_token_cache_path
            or config.train_data_path.with_suffix(".tokens.pt")
        )
        valid_cache_path = (
            config.valid_token_cache_path
            or config.valid_data_path.with_suffix(".tokens.pt")
        )

        train_data = load_token_data(
            data_path=config.train_data_path,
            token_cache_path=train_cache_path,
            tokenizer=tokenizer,
            rebuild_token_cache=config.rebuild_token_cache,
        )
        valid_data = load_token_data(
            data_path=config.valid_data_path,
            token_cache_path=valid_cache_path,
            tokenizer=tokenizer,
            rebuild_token_cache=config.rebuild_token_cache,
        )
        return train_data, valid_data

    data = load_token_data(
        data_path=config.data_path,
        token_cache_path=config.token_cache_path,
        tokenizer=tokenizer,
        rebuild_token_cache=config.rebuild_token_cache,
    )
    return split_data(data, config.train_split)


@torch.no_grad()
def estimate_loss(
    model: TransformerLM,
    train_data: torch.Tensor,
    valid_data: torch.Tensor,
    batch_size: int,
    context_length: int,
    eval_iters: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    result = {}

    for split_name, data in (("train", train_data), ("valid", valid_data)):
        losses = []

        for _ in range(eval_iters):
            x, y = get_batch(data, batch_size, context_length, device)
            logits = model(x)
            loss = cross_entropy_loss(logits, y)
            losses.append(loss.item())

        result[split_name] = sum(losses) / len(losses)

    model.train()
    return result


def save_checkpoint(
    path: Path,
    model: TransformerLM,
    optimizer: torch.optim.Optimizer,
    model_config: TransformerConfig,
    train_config: TrainConfig,
    step: int,
    loss: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "loss": loss,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_config": asdict(model_config),
            "train_config": config_to_dict(train_config),
        },
        path,
    )


def main() -> None:
    config = parse_args()
    device = resolve_device(config.device)

    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
        torch.set_float32_matmul_precision("high")

    tokenizer = Tokenizer.from_files(
        vocab_path=config.vocab_path,
        merges_path=config.merges_path,
        special_tokens=["<|endoftext|>"],
    )

    train_data, valid_data = load_train_valid_data(config, tokenizer)

    vocab_size = config.vocab_size if config.vocab_size > 0 else len(tokenizer.vocab)
    model_config = TransformerConfig(
        vocab_size=vocab_size,
        context_length=config.context_length,
        d_model=config.d_model,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        d_ff=config.d_ff,
        rope_theta=config.rope_theta,
        rms_norm_eps=config.rms_norm_eps,
    )

    model = TransformerLM(model_config, device=device).to(device)

    optimizer = configure_adamw_for_model(
        model,
        AdamWConfig(
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            beta1=config.beta1,
            beta2=config.beta2,
            eps=config.eps,
        ),
    )

    lr_schedule = CosineLRScheduleConfig(
        max_lr=config.learning_rate,
        min_lr=config.min_lr,
        warmup_steps=config.warmup_steps,
        max_steps=config.max_steps,
    )

    print(f"device: {device}")
    print(f"tokens: train={len(train_data):,}, valid={len(valid_data):,}")
    print(f"model parameters: {sum(p.numel() for p in model.parameters()):,}")

    start_time = time.time()
    last_loss = float("nan")

    for step in range(config.max_steps):
        lr = get_cosine_lr(step, lr_schedule)
        set_learning_rate(optimizer, lr)

        x, y = get_batch(
            data=train_data,
            batch_size=config.batch_size,
            context_length=config.context_length,
            device=device,
        )

        logits = model(x)
        loss = cross_entropy_loss(logits, y)
        last_loss = loss.item()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

        optimizer.step()

        next_step = step + 1
        should_eval = next_step == 1 or next_step % config.eval_interval == 0
        if should_eval:
            losses = estimate_loss(
                model=model,
                train_data=train_data,
                valid_data=valid_data,
                batch_size=config.batch_size,
                context_length=config.context_length,
                eval_iters=config.eval_iters,
                device=device,
            )
            elapsed = time.time() - start_time
            print(
                f"step {next_step:6d} | "
                f"train loss {losses['train']:.4f} | "
                f"valid loss {losses['valid']:.4f} | "
                f"lr {lr:.2e} | "
                f"elapsed {elapsed:.1f}s"
            )

        should_checkpoint = (
            config.checkpoint_interval > 0
            and next_step % config.checkpoint_interval == 0
        )
        if should_checkpoint:
            checkpoint_path = config.checkpoint_dir / f"step_{next_step:06d}.pt"
            save_checkpoint(
                path=checkpoint_path,
                model=model,
                optimizer=optimizer,
                model_config=model_config,
                train_config=config,
                step=next_step,
                loss=last_loss,
            )
            print(f"saved checkpoint: {checkpoint_path}")

    final_checkpoint_path = config.checkpoint_dir / "latest.pt"
    save_checkpoint(
        path=final_checkpoint_path,
        model=model,
        optimizer=optimizer,
        model_config=model_config,
        train_config=config,
        step=config.max_steps,
        loss=last_loss,
    )
    print(f"saved final checkpoint: {final_checkpoint_path}")


if __name__ == "__main__":
    main()
