import pickle
import regex as re
from functools import lru_cache
from typing import Iterable, Iterator

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens=None):
        if special_tokens is None:
            special_tokens = []

        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens

        # bytes token -> id
        self.token_to_id = {token: i for i, token in vocab.items()}

        # pair -> rank
        # rank 越小，优先级越高
        self.merges_rank = {pair: i for i, pair in enumerate(merges)}

        # special token 正则
        # 按长度从大到小排序，避免一个 special token 是另一个的前缀时匹配错误
        if len(special_tokens) > 0:
            special_tokens_sorted = sorted(special_tokens, key=len, reverse=True)
            escaped = [re.escape(tok) for tok in special_tokens_sorted]
            self.special_pattern = "(" + "|".join(escaped) + ")"
        else:
            self.special_pattern = None

    @classmethod
    def from_files(cls, vocab_path, merges_path, special_tokens=None):
        with open(vocab_path, "rb") as f:
            loaded_vocab = pickle.load(f)

        with open(merges_path, "rb") as f:
            merges = pickle.load(f)

        if isinstance(loaded_vocab, dict):
            vocab = loaded_vocab
        else:
            vocab = {i: token for i, token in enumerate(loaded_vocab)}

        return cls(
            vocab=vocab,
            merges=merges,
            special_tokens=special_tokens,
        )

    def decode(self, ids: list[int]) -> str:
        byte_list = []

        for i in ids:
            byte_list.append(self.vocab[i])

        big_bytes = b"".join(byte_list)
        return big_bytes.decode("utf-8", errors="replace")

    @lru_cache(maxsize=None)
    def _bpe_encode_bytes(self, token_bytes: bytes) -> tuple[bytes, ...]:
        """
        对一个 pre-token 的 bytes 执行 BPE 合并。

        例如：
        b"Hello"
        初始拆成：
        (b"H", b"e", b"l", b"l", b"o")

        然后根据 merges_rank 不断合并。
        """

        # 初始状态：每个 byte 单独作为一个 token
        parts = tuple(bytes([b]) for b in token_bytes)

        if len(parts) <= 1:
            return parts

        while True:
            best_pair = None
            best_rank = float("inf")

            # 找当前序列中 rank 最小的相邻 pair
            for i in range(len(parts) - 1):
                pair = (parts[i], parts[i + 1])
                rank = self.merges_rank.get(pair)

                if rank is not None and rank < best_rank:
                    best_rank = rank
                    best_pair = pair

            # 如果没有任何 pair 可以合并，停止
            if best_pair is None:
                break

            # 合并当前所有不重叠的 best_pair
            new_parts = []
            i = 0

            while i < len(parts):
                if (
                    i < len(parts) - 1
                    and parts[i] == best_pair[0]
                    and parts[i + 1] == best_pair[1]
                ):
                    new_parts.append(parts[i] + parts[i + 1])
                    i += 2
                else:
                    new_parts.append(parts[i])
                    i += 1

            parts = tuple(new_parts)

            if len(parts) <= 1:
                break

        return parts

    def _encode_ordinary_text(self, text: str) -> list[int]:
        """
        encode 普通文本，不处理 special token。
        """
        ids = []

        # 先用 PAT 做 pre-tokenization
        for match in re.finditer(PAT, text):
            pre_token = match.group(0)

            # str -> UTF-8 bytes
            token_bytes = pre_token.encode("utf-8")

            # bytes -> BPE tokens
            bpe_tokens = self._bpe_encode_bytes(token_bytes)

            # BPE tokens -> ids
            for token in bpe_tokens:
                if token not in self.token_to_id:
                    raise KeyError(f"Token {token!r} not found in vocab")

                ids.append(self.token_to_id[token])

        return ids

    def encode(self, text: str) -> list[int]:
        """
        完整 encode：

        str
        -> 按 special token 切分
        -> 普通文本用 PAT 切分
        -> UTF-8 bytes
        -> BPE merge
        -> token ids
        """

        # 没有 special token 时，直接普通 encode
        if self.special_pattern is None:
            return self._encode_ordinary_text(text)

        ids = []

        # 用 special token 把文本切开
        chunks = re.split(self.special_pattern, text)

        for chunk in chunks:
            if chunk == "":
                continue

            # 如果 chunk 是 special token，直接整体转 id
            if chunk in self.special_tokens:
                special_bytes = chunk.encode("utf-8")

                if special_bytes not in self.token_to_id:
                    raise KeyError(
                        f"Special token {chunk!r} is not in vocab. "
                        f"Make sure it was added during BPE training."
                    )

                ids.append(self.token_to_id[special_bytes])

            # 否则按普通文本处理
            else:
                ids.extend(self._encode_ordinary_text(chunk))

        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        对一个字符串迭代器进行 encode。

        例如：
        tokenizer.encode_iterable(open("data.txt", encoding="utf-8"))

        这个函数会逐段 encode，然后逐个 yield token id。
        """
        for text in iterable:
            ids = self.encode(text)
            for token_id in ids:
                yield token_id


# Simple smoke test for direct script execution.
if __name__ == "__main__":
    tokenizer = Tokenizer.from_files(
        vocab_path="D:/CS336/CS336-Learn/01_bpe/vocab.bin",
        merges_path="D:/CS336/CS336-Learn/01_bpe/merges.bin",
        special_tokens=["<|endoftext|>"],
    )

    text = "我爱中国"
    ids = tokenizer.encode(text)

    print(ids)
    print(tokenizer.decode(ids))
