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


def train_bpe_tinystories(
    input_path: str | Path,
    output_dir: str | Path = "artifacts/tinystories",
    profile: bool = False,
):
    input_path = Path(input_path)
    output_dir = Path(output_dir)

    if not input_path.is_file():
        raise FileNotFoundError(
            f"Training file not found: {input_path}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    special_tokens = ["<|endoftext|>"]
    profiler = cProfile.Profile() if profile else None

    # 训练；计时不包含保存文件
    if profiler is not None:
        profiler.enable()

    started = time.perf_counter()

    try:
        vocab, merges = train_bpe(
            input_path=input_path,
            vocab_size=10_000,
            special_tokens=special_tokens,
        )
        elapsed_seconds = time.perf_counter() - started
    finally:
        if profiler is not None:
            profiler.disable()

    # 当前进程的峰值 RSS，包含导入和训练
    # macOS 返回 bytes；Linux/Colab 返回 KiB
    peak_rss = resource.getrusage(
        resource.RUSAGE_SELF
    ).ru_maxrss

    peak_rss_bytes = (
        peak_rss
        if sys.platform == "darwin"
        else peak_rss * 1024
    )

    # 保留原始 dict[int, bytes] 和 list[tuple[bytes, bytes]]
    with (output_dir / "vocab.pkl").open("wb") as stream:
        pickle.dump(
            vocab, stream, protocol=pickle.HIGHEST_PROTOCOL
        )

    with (output_dir / "merges.pkl").open("wb") as stream:
        pickle.dump(
            merges, stream, protocol=pickle.HIGHEST_PROTOCOL
        )

    # 按字节长度查找最长 token
    longest_id, longest_token = max(
        vocab.items(),
        key=lambda item: len(item[1]),
    )

    # 同时报告排除特殊 token 后的最长 token
    special_bytes = {
        token.encode("utf-8") for token in special_tokens
    }

    ordinary_id, ordinary_token = max(
        (
            item
            for item in vocab.items()
            if item[1] not in special_bytes
        ),
        key=lambda item: len(item[1]),
    )

    report = {
        "input_path": str(input_path),
        "vocab_size": len(vocab),
        "num_merges": len(merges),
        "elapsed_seconds": elapsed_seconds,
        "peak_process_rss_gib": peak_rss_bytes / (1024 ** 3),
        "profile_enabled": profile,
        "longest_token_id": longest_id,
        "longest_token_bytes": len(longest_token),
        "longest_token_repr": repr(longest_token),
        "longest_ordinary_token_id": ordinary_id,
        "longest_ordinary_token_bytes": len(ordinary_token),
        "longest_ordinary_token_repr": repr(ordinary_token),
    }

    # Profiling：累计耗时最高的函数，以及两个训练阶段的耗时
    if profiler is not None:
        profiler.dump_stats(
            str(output_dir / "training.prof")
        )

        buffer = io.StringIO()
        stats = pstats.Stats(profiler, stream=buffer)

        stage_seconds = {}

        for (_, _, function_name), values in stats.stats.items():
            if function_name in {
                "_count_pretokens",
                "_train_from_counts",
            }:
                stage_seconds[function_name] = (
                    stage_seconds.get(function_name, 0.0)
                    + values[3]
                )

        report["stage_cumulative_seconds"] = stage_seconds

        if stage_seconds:
            report["slowest_stage"] = max(
                stage_seconds,
                key=stage_seconds.get,
            )

        stats.strip_dirs().sort_stats(
            "cumulative"
        ).print_stats(20)

        profile_text = buffer.getvalue()

        (output_dir / "profile.txt").write_text(
            profile_text,
            encoding="utf-8",
        )

        print(profile_text)

    report_text = json.dumps(
        report, ensure_ascii=False, indent=2
    )

    (output_dir / "report.json").write_text(
        report_text,
        encoding="utf-8",
    )

    print(report_text)
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
        default=Path("artifacts/tinystories"),
    )
    parser.add_argument(
        "--profile", action="store_true"
    )

    args = parser.parse_args()

    train_bpe_tinystories(
        args.input,
        args.output,
        profile=args.profile,
    )