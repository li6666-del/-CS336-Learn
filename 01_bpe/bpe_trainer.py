import argparse
import pickle
from pathlib import Path

import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def read_text(input_path: str | Path, max_bytes: int | None = None) -> str:
    if max_bytes is None:
        return Path(input_path).read_text(encoding="utf-8")

    with open(input_path, "rb") as f:
        raw_bytes = f.read(max_bytes)

    return raw_bytes.decode("utf-8", errors="ignore")


def train_bpe(input_path, vocab_size, special_tokens, max_bytes=None):
    text = read_text(input_path, max_bytes=max_bytes)

    special_pattern = "|".join(re.escape(t) for t in special_tokens)
    pattern = f"({special_pattern})" if special_pattern else None

    if pattern:
        text_list = re.split(pattern, text)
    else:
        text_list = [text]

    word_freq = {}
    for chunk in text_list:
        if chunk in special_tokens:
            continue
        for match in re.finditer(PAT, chunk):
            word = match.group().encode("utf-8")
            key = tuple(bytes([b]) for b in word)
            word_freq[key] = word_freq.get(key, 0) + 1

    vocab = {}
    for i in range(256):
        vocab[i] = bytes([i])

    next_idx = 256
    for token in special_tokens:
        vocab[next_idx] = token.encode("utf-8")
        next_idx += 1

    merges = []
    target_merges = vocab_size - len(vocab)
    while len(vocab) < vocab_size:
        if len(merges) % 50 == 0:
            print(f"merge progress: {len(merges)}/{target_merges}")
        pair_freq = {}
        for word, freq in word_freq.items():
            for i in range(len(word) - 1):
                key = (word[i], word[i + 1])
                pair_freq[key] = pair_freq.get(key, 0) + freq

        if not pair_freq:
            break

        best_pair = max(pair_freq, key=lambda x: (pair_freq[x], x))
        merged_token = best_pair[0] + best_pair[1]

        new_word_freq = {}
        for word, freq in word_freq.items():
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == best_pair[0] and word[i + 1] == best_pair[1]:
                    new_word.append(merged_token)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            new_word_freq[new_word] = new_word_freq.get(new_word, 0) + freq
        word_freq = new_word_freq

        vocab[len(vocab)] = merged_token
        merges.append(best_pair)

    return vocab, merges


def parse_args():
    parser = argparse.ArgumentParser(description="Train a byte-level BPE tokenizer.")

    parser.add_argument(
        "--input_path",
        type=Path,
        default=Path("data/TinyStoriesV2-GPT4-valid.txt"),
    )
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument(
        "--special_tokens",
        nargs="*",
        default=["<|endoftext|>"],
    )
    parser.add_argument(
        "--max_bytes",
        type=int,
        default=None,
        help="Only read this many bytes from input_path. Useful for large corpora.",
    )
    parser.add_argument(
        "--vocab_path",
        type=Path,
        default=Path("01_bpe/vocab.bin"),
    )
    parser.add_argument(
        "--merges_path",
        type=Path,
        default=Path("01_bpe/merges.bin"),
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    vocab, merges = train_bpe(
        input_path=args.input_path,
        vocab_size=args.vocab_size,
        special_tokens=args.special_tokens,
        max_bytes=args.max_bytes,
    )

    vocab_list = [None] * len(vocab)
    for idx, token_bytes in vocab.items():
        vocab_list[idx] = token_bytes

    args.vocab_path.parent.mkdir(parents=True, exist_ok=True)
    args.merges_path.parent.mkdir(parents=True, exist_ok=True)

    with open(args.vocab_path, "wb") as f:
        pickle.dump(vocab_list, f)

    with open(args.merges_path, "wb") as f:
        pickle.dump(merges, f)

    print(f"vocab size: {len(vocab)}, merges count: {len(merges)}")
