from __future__ import annotations

import heapq
import mmap
import os
import re
from collections import Counter, defaultdict

import regex


Pair = tuple[bytes, bytes]

BYTE_TOKENS = tuple(bytes([i]) for i in range(256))

PRETOKEN_PATTERN = regex.compile(
    r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
)


class _PairCandidate:
    """优先选择频率较高的 pair；同频时选择字典序较大的 pair。"""

    __slots__ = ("count", "pair")

    def __init__(self, count: int, pair: Pair):
        self.count = count
        self.pair = pair

    def __lt__(self, other: _PairCandidate) -> bool:
        if self.count != other.count:
            return self.count > other.count
        return self.pair > other.pair


def _count_pretokens(
    input_path: str | os.PathLike,
    special_tokens: list[str],
) -> Counter[bytes]:
    """特殊 token 隔断文本，各段分别进行 GPT-2 预分词。"""

    counts: Counter[bytes] = Counter()
    special_pattern = None

    if special_tokens:
        alternatives = sorted(
            (token.encode("utf-8") for token in special_tokens),
            key=len,
            reverse=True,
        )
        special_pattern = re.compile(
            b"|".join(re.escape(token) for token in alternatives)
        )

    def count_segment(segment: bytes) -> None:
        text = segment.decode("utf-8")
        counts.update(
            match.group().encode("utf-8")
            for match in PRETOKEN_PATTERN.finditer(text)
        )

    with open(input_path, "rb") as stream:
        if os.fstat(stream.fileno()).st_size == 0:
            return counts

        with mmap.mmap(
            stream.fileno(), 0, access=mmap.ACCESS_READ
        ) as data:
            start = 0

            if special_pattern is not None:
                for match in special_pattern.finditer(data):
                    count_segment(data[start:match.start()])
                    start = match.end()

            count_segment(data[start:])

    return counts


def _merge_word(
    word: tuple[bytes, ...],
    pair: Pair,
) -> tuple[bytes, ...]:
    """从左向右合并，避免重叠位置被重复使用。"""

    first, second = pair
    merged_token = first + second
    result: list[bytes] = []
    i = 0

    while i < len(word):
        if (
            i + 1 < len(word)
            and word[i] == first
            and word[i + 1] == second
        ):
            result.append(merged_token)
            i += 2
        else:
            result.append(word[i])
            i += 1

    return tuple(result)


def _train_from_counts(
    counts: Counter[bytes],
    vocab: dict[int, bytes],
    vocab_size: int,
) -> tuple[dict[int, bytes], list[Pair]]:

    words = [
        tuple(BYTE_TOKENS[b] for b in raw)
        for raw in counts
    ]
    frequencies = list(counts.values())

    pair_counts: Counter[Pair] = Counter()

    # pair -> 包含这个 pair 的预分词编号
    pair_words: dict[Pair, set[int]] = defaultdict(set)

    for word_id, word in enumerate(words):
        local_counts = Counter(zip(word, word[1:]))

        for pair, occurrences in local_counts.items():
            pair_counts[pair] += (
                occurrences * frequencies[word_id]
            )
            pair_words[pair].add(word_id)

    heap = [
        _PairCandidate(count, pair)
        for pair, count in pair_counts.items()
    ]
    heapq.heapify(heap)

    merges: list[Pair] = []

    while len(vocab) < vocab_size and heap:
        candidate = heapq.heappop(heap)
        best_pair = candidate.pair

        # 忽略频率已过期的堆记录
        if candidate.count != pair_counts.get(best_pair, 0):
            continue

        merges.append(best_pair)
        vocab[len(vocab)] = best_pair[0] + best_pair[1]

        changed_pairs: set[Pair] = set()

        # 只处理包含 best_pair 的预分词
        for word_id in tuple(pair_words[best_pair]):
            old_word = words[word_id]
            new_word = _merge_word(old_word, best_pair)

            old_counts = Counter(zip(old_word, old_word[1:]))
            new_counts = Counter(zip(new_word, new_word[1:]))
            frequency = frequencies[word_id]

            for pair in old_counts.keys() | new_counts.keys():
                difference = new_counts[pair] - old_counts[pair]

                if difference:
                    pair_counts[pair] += difference * frequency
                    changed_pairs.add(pair)

                if not new_counts[pair]:
                    pair_words[pair].discard(word_id)
                elif not old_counts[pair]:
                    pair_words[pair].add(word_id)

            words[word_id] = new_word

        # 更新发生变化的 pair
        for pair in changed_pairs:
            count = pair_counts[pair]

            if count > 0:
                heapq.heappush(
                    heap, _PairCandidate(count, pair)
                )
            else:
                del pair_counts[pair]
                pair_words.pop(pair, None)

        # 定期清理过期记录，限制堆的大小
        if len(heap) > 4 * max(len(pair_counts), 1):
            heap = [
                _PairCandidate(count, pair)
                for pair, count in pair_counts.items()
            ]
            heapq.heapify(heap)

    return vocab, merges


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    if not isinstance(vocab_size, int) or isinstance(vocab_size, bool):
        raise TypeError("vocab_size must be an integer")

    if any(not isinstance(token, str) for token in special_tokens):
        raise TypeError("special_tokens must contain strings")

    if any(not token for token in special_tokens):
        raise ValueError("special_tokens cannot contain an empty string")

    # 去除重复的特殊 token，保留输入顺序
    special_tokens = list(dict.fromkeys(special_tokens))

    # ID 0–255：全部单字节 token
    vocab = dict(enumerate(BYTE_TOKENS))

    # 接着加入特殊 token
    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")

    if vocab_size < len(vocab):
        raise ValueError(
            f"vocab_size must be at least {len(vocab)} "
            "to include all bytes and special tokens"
        )

    counts = _count_pretokens(input_path, special_tokens)

    if vocab_size == len(vocab):
        return vocab, []

    return _train_from_counts(counts, vocab, vocab_size)