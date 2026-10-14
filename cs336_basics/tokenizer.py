from __future__ import annotations

import pickle
from collections import deque
from collections.abc import Iterable, Iterator
from pathlib import Path

import regex


PRETOKEN_PATTERN = regex.compile(
    r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
)

BYTE_TOKENS = tuple(bytes([i]) for i in range(256))


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = dict(vocab)
        self.merges = list(merges)
        self.special_tokens = list(
            dict.fromkeys(special_tokens or [])
        )

        if any(not token for token in self.special_tokens):
            raise ValueError(
                "Special tokens cannot be empty strings"
            )

        self.token_to_id = {
            token: token_id
            for token_id, token in self.vocab.items()
        }

        # 为缺失的特殊 token 分配新的 ID
        next_id = max(self.vocab, default=-1) + 1
        self._special_ids: dict[str, int] = {}

        for token in self.special_tokens:
            raw = token.encode("utf-8")

            if raw not in self.token_to_id:
                self.vocab[next_id] = raw
                self.token_to_id[raw] = next_id
                next_id += 1

            self._special_ids[token] = self.token_to_id[raw]

        # 越早创建的 merge，优先级越高
        self.merge_ranks: dict[
            tuple[bytes, bytes], int
        ] = {}

        for rank, pair in enumerate(self.merges):
            self.merge_ranks.setdefault(pair, rank)

        # 较长的特殊 token 优先匹配
        self._special_pattern = None

        if self.special_tokens:
            ordered = sorted(
                self.special_tokens,
                key=len,
                reverse=True,
            )

            self._special_pattern = regex.compile(
                "|".join(
                    regex.escape(token)
                    for token in ordered
                )
            )

        self._special_lookahead = max(
            map(len, self.special_tokens),
            default=0,
        )

        self._cache: dict[bytes, tuple[int, ...]] = {}

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str | Path,
        merges_filepath: str | Path,
        special_tokens: list[str] | None = None,
    ) -> Tokenizer:
        # 读取前面训练脚本保存的 .pkl 文件
        with open(vocab_filepath, "rb") as stream:
            vocab = pickle.load(stream)

        with open(merges_filepath, "rb") as stream:
            merges = pickle.load(stream)

        return cls(
            vocab,
            merges,
            special_tokens=special_tokens,
        )

    def _iter_units(self, text: str):
        """返回 start、end、special_id；普通预分词使用 None。"""

        start = 0

        if self._special_pattern is not None:
            for special in self._special_pattern.finditer(text):
                for match in PRETOKEN_PATTERN.finditer(
                    text, start, special.start()
                ):
                    yield match.start(), match.end(), None

                yield (
                    special.start(),
                    special.end(),
                    self._special_ids[special.group()],
                )

                start = special.end()

        for match in PRETOKEN_PATTERN.finditer(
            text, start, len(text)
        ):
            yield match.start(), match.end(), None

    def _encode_pretoken(self, raw: bytes) -> tuple[int, ...]:
        cached = self._cache.get(raw)

        if cached is not None:
            return cached

        tokens = [BYTE_TOKENS[b] for b in raw]

        while len(tokens) > 1:
            best_pair = None
            best_rank = float("inf")

            # 在当前相邻 pair 中选 merge rank 最小的
            for pair in zip(tokens, tokens[1:]):
                rank = self.merge_ranks.get(pair)

                if rank is not None and rank < best_rank:
                    best_pair = pair
                    best_rank = rank

            if best_pair is None:
                break

            first, second = best_pair
            merged = []
            i = 0

            # 从左向右合并该 pair 的所有非重叠出现位置
            while i < len(tokens):
                if (
                    i + 1 < len(tokens)
                    and tokens[i] == first
                    and tokens[i + 1] == second
                ):
                    merged.append(first + second)
                    i += 2
                else:
                    merged.append(tokens[i])
                    i += 1

            tokens = merged

        result = tuple(
            self.token_to_id[token] for token in tokens
        )

        # 限制缓存大小，避免随输入文件持续增长
        if len(raw) <= 128:
            if len(self._cache) >= 512:
                self._cache.clear()

            self._cache[raw] = result

        return result

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []

        for start, end, special_id in self._iter_units(text):
            if special_id is not None:
                ids.append(special_id)
            else:
                raw = text[start:end].encode("utf-8")
                ids.extend(self._encode_pretoken(raw))

        return ids

    def encode_iterable(
        self,
        iterable: Iterable[str],
    ) -> Iterator[int]:
        """懒加载输入，并保留跨 chunk 的分词上下文。"""

        buffer = ""

        chunks = (
            text[start:start + 8192]
            for text in iterable
            for start in range(0, len(text), 8192)
        )

        for chunk in chunks:
            buffer += chunk

            safe_end = (
                len(buffer) - self._special_lookahead
            )

            pending = deque()
            consumed = 0

            for unit in self._iter_units(buffer):
                pending.append(unit)

                # 保留特殊 token 前缀、相邻空白和缩写上下文
                if (
                    len(pending)
                    <= self._special_lookahead + 2
                ):
                    continue

                start, end, special_id = pending.popleft()

                if end > safe_end:
                    break

                if special_id is not None:
                    yield special_id
                else:
                    raw = buffer[start:end].encode("utf-8")
                    yield from self._encode_pretoken(raw)

                consumed = end

            buffer = buffer[consumed:]

        yield from self.encode(buffer)

    def decode(self, ids: Iterable[int]) -> str:
        raw = b"".join(
            self.vocab[token_id] for token_id in ids
        )

        return raw.decode("utf-8", errors="replace")