import argparse
import sys
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "01_bpe"))

from bpe_tokenizer import Tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Encode a text file into token ids.")

    parser.add_argument("--input_path", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
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
    parser.add_argument(
        "--dtype",
        choices=["int32", "int64"],
        default="int32",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start_time = time.time()

    tokenizer = Tokenizer.from_files(
        vocab_path=args.vocab_path,
        merges_path=args.merges_path,
        special_tokens=["<|endoftext|>"],
    )

    print(f"reading: {args.input_path}", flush=True)
    text = args.input_path.read_text(encoding="utf-8")

    print("encoding...", flush=True)
    token_ids = tokenizer.encode(text)

    dtype = torch.int32 if args.dtype == "int32" else torch.long
    tokens = torch.tensor(token_ids, dtype=dtype)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tokens, args.output_path)

    elapsed = time.time() - start_time
    print(f"saved: {args.output_path}", flush=True)
    print(f"tokens: {len(tokens):,}", flush=True)
    print(f"dtype: {tokens.dtype}", flush=True)
    print(f"elapsed: {elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
