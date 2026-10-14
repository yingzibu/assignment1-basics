from __future__ import annotations

import argparse
import cProfile
import io
import json
import pickle
import pstats
import resource
import sys
import time
from pathlib import Path

from cs336_basics.train_bpe import train_bpe


SPECIAL_TOKENS = ["<|endoftext|>"]


def summarize_vocab(vocab: dict[int, bytes]) -> dict:
    special_bytes = {
        token.encode("utf-8") for token in SPECIAL_TOKENS
    }

    ordinary = [
        token
        for token in vocab.values()
        if token not in special_bytes
    ]

    longest = max(vocab.values(), key=len)

    ordered = sorted(
        ordinary,
        key=lambda token: (len(token), token),
        reverse=True,
    )

    return {
        "vocab_size": len(vocab),
        "longest_token_bytes": len(longest),
        "longest_token_repr": repr(longest),
        "longest_ordinary_token_bytes": len(ordered[0]),
        "longest_ordinary_token_repr": repr(ordered[0]),
        "mean_ordinary_token_bytes": (
            sum(map(len, ordinary)) / len(ordinary)
        ),
        "longest_ordinary_examples": [
            repr(token) for token in ordered[:20]
        ],
    }


def compare_vocabularies(tinystories_vocab, owt_vocab) -> dict:
    # 比较学习得到的 token，排除初始字节及特殊 token
    excluded = {bytes([i]) for i in range(256)}
    excluded.update(
        token.encode("utf-8") for token in SPECIAL_TOKENS
    )

    tiny_tokens = set(tinystories_vocab.values()) - excluded
    owt_tokens = set(owt_vocab.values()) - excluded

    shared = tiny_tokens & owt_tokens
    union = tiny_tokens | owt_tokens

    def longest_examples(tokens):
        ordered = sorted(
            tokens,
            key=lambda token: (len(token), token),
            reverse=True,
        )
        return [repr(token) for token in ordered[:20]]

    return {
        "tinystories": summarize_vocab(tinystories_vocab),
        "openwebtext": summarize_vocab(owt_vocab),
        "shared_learned_tokens": len(shared),
        "tinystories_only_learned_tokens": len(
            tiny_tokens - owt_tokens
        ),
        "owt_only_learned_tokens": len(
            owt_tokens - tiny_tokens
        ),
        "learned_token_jaccard": (
            len(shared) / len(union) if union else 1.0
        ),
        "tinystories_only_longest_examples": longest_examples(
            tiny_tokens - owt_tokens
        ),
        "owt_only_longest_examples": longest_examples(
            owt_tokens - tiny_tokens
        ),
        "interpretation_note": (
            "Differences reflect both corpus "
            "and vocabulary-size limits."
        ),
    }


def train_bpe_expts_owt(
    input_path: str | Path,
    output_dir: str | Path = "artifacts/owt",
    tinystories_vocab_path: str | Path | None = None,
    profile: bool = False,
):
    input_path = Path(input_path)
    output_dir = Path(output_dir)

    if not input_path.is_file():
        raise FileNotFoundError(
            f"Training file not found: {input_path}"
        )

    # 提前检查 TinyStories 词表，避免训练结束才发现路径错误
    tiny_vocab = None

    if tinystories_vocab_path is not None:
        with Path(tinystories_vocab_path).open("rb") as stream:
            tiny_vocab = pickle.load(stream)

    output_dir.mkdir(parents=True, exist_ok=True)

    profiler = cProfile.Profile() if profile else None

    if profiler is not None:
        profiler.enable()

    started = time.perf_counter()

    try:
        vocab, merges = train_bpe(
            input_path=input_path,
            vocab_size=32_000,
            special_tokens=SPECIAL_TOKENS,
        )
        elapsed_seconds = time.perf_counter() - started
    finally:
        if profiler is not None:
            profiler.disable()

    # 当前进程的峰值 RSS；计量单位随系统转换
    peak_rss = resource.getrusage(
        resource.RUSAGE_SELF
    ).ru_maxrss

    peak_rss_bytes = (
        peak_rss
        if sys.platform == "darwin"
        else peak_rss * 1024
    )

    with (output_dir / "vocab.pkl").open("wb") as stream:
        pickle.dump(
            vocab, stream, protocol=pickle.HIGHEST_PROTOCOL
        )

    with (output_dir / "merges.pkl").open("wb") as stream:
        pickle.dump(
            merges, stream, protocol=pickle.HIGHEST_PROTOCOL
        )

    report = summarize_vocab(vocab)

    report.update({
        "input_path": str(input_path),
        "num_merges": len(merges),
        "elapsed_seconds": elapsed_seconds,
        "peak_process_rss_gib": peak_rss_bytes / (1024 ** 3),
        "profile_enabled": profile,
    })

    report_text = json.dumps(
        report, ensure_ascii=False, indent=2
    )

    (output_dir / "report.json").write_text(
        report_text, encoding="utf-8"
    )

    print(report_text)

    # (b) 比较两个词表
    if tiny_vocab is not None:
        comparison = compare_vocabularies(
            tiny_vocab, vocab
        )

        comparison_text = json.dumps(
            comparison, ensure_ascii=False, indent=2
        )

        (output_dir / "comparison.json").write_text(
            comparison_text, encoding="utf-8"
        )

        print(comparison_text)

    if profiler is not None:
        profiler.dump_stats(
            str(output_dir / "training.prof")
        )

        buffer = io.StringIO()

        stats = pstats.Stats(
            profiler, stream=buffer
        ).strip_dirs()

        stats.sort_stats("cumulative").print_stats(20)
        stats.sort_stats("tottime").print_stats(20)

        (output_dir / "profile.txt").write_text(
            buffer.getvalue(), encoding="utf-8"
        )

        print(buffer.getvalue())

    print(f"Saved results to: {output_dir.resolve()}")

    return vocab, merges, report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input", type=Path, required=True
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/owt"),
    )
    parser.add_argument(
        "--tinystories-vocab", type=Path
    )
    parser.add_argument(
        "--profile", action="store_true"
    )

    args = parser.parse_args()

    train_bpe_expts_owt(
        args.input,
        args.output,
        tinystories_vocab_path=args.tinystories_vocab,
        profile=args.profile,
    )