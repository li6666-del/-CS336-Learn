import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def train_bpe(input_path, vocab_size, special_tokens):
    with open(input_path, "r", encoding="UTF-8") as f:
        text = f.read()

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


if __name__ == "__main__":
    input_path = "D:/CS336/CS336-Learn/data/TinyStoriesV2-GPT4-valid.txt"
    vocab_size = 10000
    special_tokens = ["<|endoftext|>"]

    vocab, merges = train_bpe(input_path, vocab_size, special_tokens)

    vocab_list = [None] * len(vocab)
    for idx, token_bytes in vocab.items():
        vocab_list[idx] = token_bytes

    with open("D:/CS336/CS336-Learn/01_bpe/vocab.bin", "wb") as f:
        import pickle
        pickle.dump(vocab_list, f)

    with open("D:/CS336/CS336-Learn/01_bpe/merges.bin", "wb") as f:
        import pickle
        pickle.dump(merges, f)

    print(f"vocab size: {len(vocab)}, merges count: {len(merges)}")
