import argparse
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "01_bpe"))
sys.path.insert(0, str(ROOT / "02_model"))

from bpe_tokenizer import Tokenizer
from transformer_made import TransformerConfig, TransformerLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample text from a trained Transformer LM.")

    parser.add_argument(
        "--checkpoint_path",
        type=Path,
        default=ROOT / "checkpoints" / "step_004000.pt",
    )
    parser.add_argument(
        "--vocab_path",
        type=Path,
        default=ROOT / "01_bpe" / "vocab.bin",
    )
    parser.add_argument(
        "--merges_path",
        type=Path,
        default=ROOT / "01_bpe" / "merges.bin",
    )
    parser.add_argument("--prompt", type=str, default="Once upon a time")
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", type=str, default="auto")

    return parser.parse_args()


def resolve_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def load_model(checkpoint_path: Path, device: torch.device) -> TransformerLM:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_config = TransformerConfig(**checkpoint["model_config"])

    model = TransformerLM(model_config, device=device).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model


@torch.no_grad()
def generate(
    model: TransformerLM,
    token_ids: list[int],
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    device: torch.device,
) -> list[int]:
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")

    ids = torch.tensor(token_ids, dtype=torch.long, device=device)[None, :]

    for _ in range(max_new_tokens):
        context = ids[:, -model.config.context_length :]
        logits = model(context)
        next_token_logits = logits[:, -1, :] / temperature

        if top_k is not None and top_k > 0:
            top_k = min(top_k, next_token_logits.size(-1))
            values, _ = torch.topk(next_token_logits, top_k)
            cutoff = values[:, [-1]]
            next_token_logits = next_token_logits.masked_fill(
                next_token_logits < cutoff,
                float("-inf"),
            )

        probs = torch.softmax(next_token_logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        ids = torch.cat((ids, next_id), dim=1)

    return ids[0].tolist()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision("high")

    tokenizer = Tokenizer.from_files(
        vocab_path=args.vocab_path,
        merges_path=args.merges_path,
        special_tokens=["<|endoftext|>"],
    )
    model = load_model(args.checkpoint_path, device)

    prompt_ids = tokenizer.encode(args.prompt)
    output_ids = generate(
        model=model,
        token_ids=prompt_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        device=device,
    )

    print(tokenizer.decode(output_ids))


if __name__ == "__main__":
    main()
