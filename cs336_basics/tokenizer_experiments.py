from __future__ import annotations

import argparse
import json
import random
import struct
import time
from pathlib import Path

import numpy as np

from cs336_basics.tokenizer import Tokenizer


EOS = "<|endoftext|>"
PILE_BYTES = 825 * 10**9
HEADER_BYTES = 256


def save_json(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def iter_documents(path, chunk_chars=1 << 20):
    """保留正文及空白，去除文档分隔符。"""
    pending = ""

    with open(path, encoding="utf-8", newline="") as stream:
        while chunk := stream.read(chunk_chars):
            parts = (pending + chunk).split(EOS)
            pending = parts.pop()

            for document in parts:
                if document.strip():
                    yield document

    if pending.strip():
        yield pending


def sample_documents(path, n=10, seed=42):
    """Reservoir sampling：扫描全文，均匀抽取 n 篇文档。"""
    rng = random.Random(seed)
    samples = []
    seen = 0

    for index, text in enumerate(iter_documents(path)):
        seen += 1
        sample = {
            "document_index": index,
            "text": text,
        }

        if len(samples) < n:
            samples.append(sample)
        else:
            slot = rng.randrange(seen)
            if slot < n:
                samples[slot] = sample

    if len(samples) < n:
        raise ValueError(
            f"Need {n} nonempty documents; "
            f"found {seen} in {path}"
        )

    return samples, seen


def measure(tokenizer, samples, name, output_dir):
    sizes = [
        len(sample["text"].encode("utf-8"))
        for sample in samples
    ]

    # 仅计时编码过程
    started = time.perf_counter()
    encoded = [
        tokenizer.encode(sample["text"])
        for sample in samples
    ]
    elapsed = time.perf_counter() - started

    n_bytes = sum(sizes)
    n_tokens = sum(map(len, encoded))
    throughput = n_bytes / elapsed

    save_json(
        output_dir / f"{name}_sample_ids.json",
        {
            "document_indices": [
                sample["document_index"] for sample in samples
            ],
            "utf8_bytes_per_document": sizes,
            "ids": encoded,
        },
    )

    return {
        "vocab_size": len(tokenizer.vocab),
        "documents": len(samples),
        "utf8_bytes": n_bytes,
        "tokens": n_tokens,
        "bytes_per_token": n_bytes / n_tokens,
        "encode_seconds": elapsed,
        "bytes_per_second": throughput,
        "pile_825gb_estimate_hours": (
            PILE_BYTES / throughput / 3600
        ),
    }


def write_npy_header(stream, n_tokens):
    """预留固定长度的 NPY header，编码完成后更新数组长度。"""
    header_length = HEADER_BYTES - 10

    header = repr({
        "descr": "<u2",
        "fortran_order": False,
        "shape": (n_tokens,),
    }).encode("ascii")

    padding = header_length - len(header) - 1
    if padding < 0:
        raise ValueError(
            "Array shape exceeds reserved NPY header space"
        )

    stream.seek(0)
    stream.write(
        b"\x93NUMPY\x01\x00"
        + struct.pack("<H", header_length)
    )
    stream.write(header + b" " * padding + b"\n")


def encode_to_npy(
    tokenizer,
    input_path,
    output_path,
    buffer_tokens=65_536,
):
    """流式编码并直接写入标准 uint16 .npy 文件。"""
    if any(
        token_id < 0 or token_id > 65_535
        for token_id in tokenizer.vocab
    ):
        raise ValueError(
            "uint16 requires every token ID "
            "to be between 0 and 65535"
        )

    if buffer_tokens < 1:
        raise ValueError("buffer_tokens must be positive")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = output_path.with_name(
        output_path.name + ".partial"
    )

    buffer = np.empty(buffer_tokens, dtype="<u2")
    used = 0
    n_tokens = 0
    started = time.perf_counter()

    with open(
        input_path, encoding="utf-8", newline=""
    ) as source:
        with partial_path.open("wb") as destination:
            write_npy_header(destination, 0)

            for token_id in tokenizer.encode_iterable(source):
                buffer[used] = token_id
                used += 1
                n_tokens += 1

                if used == buffer_tokens:
                    destination.write(buffer.tobytes())
                    used = 0

            if used:
                destination.write(buffer[:used].tobytes())

            write_npy_header(destination, n_tokens)

    partial_path.replace(output_path)

    return {
        "input": str(input_path),
        "output": str(output_path),
        "dtype": "uint16",
        "tokens": n_tokens,
        "file_bytes": output_path.stat().st_size,
        "encode_and_write_seconds": (
            time.perf_counter() - started
        ),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--tiny-train", type=Path,
        default=Path("data/TinyStoriesV2-GPT4-train.txt"),
    )
    parser.add_argument(
        "--tiny-valid", type=Path,
        default=Path("data/TinyStoriesV2-GPT4-valid.txt"),
    )
    parser.add_argument(
        "--owt-train", type=Path,
        default=Path("data/owt_train.txt"),
    )
    parser.add_argument(
        "--owt-valid", type=Path,
        default=Path("data/owt_valid.txt"),
    )
    parser.add_argument(
        "--tiny-tokenizer-dir", type=Path,
        default=Path("artifacts/tinystories_profile"),
    )
    parser.add_argument(
        "--owt-tokenizer-dir", type=Path,
        default=Path("artifacts/owt"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("artifacts/tokenizer_experiments"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--serialize",
        action="store_true",
        help="Also encode all four complete datasets",
    )

    args = parser.parse_args()

    required = [args.tiny_train, args.owt_train]

    for directory in [
        args.tiny_tokenizer_dir, args.owt_tokenizer_dir
    ]:
        required.extend([
            directory / "vocab.pkl",
            directory / "merges.pkl",
        ])

    if args.serialize:
        required.extend([args.tiny_valid, args.owt_valid])

    missing = [
        str(path) for path in required if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing inputs:\n" + "\n".join(missing)
        )

    args.output.mkdir(parents=True, exist_ok=True)

    def load_tokenizer(directory):
        return Tokenizer.from_files(
            directory / "vocab.pkl",
            directory / "merges.pkl",
            special_tokens=[EOS],
        )

    tiny = load_tokenizer(args.tiny_tokenizer_dir)
    owt = load_tokenizer(args.owt_tokenizer_dir)

    print(
        "Sampling 10 documents from each corpus; "
        "this scans both training files.",
        flush=True,
    )

    tiny_samples, tiny_seen = sample_documents(
        args.tiny_train, seed=args.seed
    )
    owt_samples, owt_seen = sample_documents(
        args.owt_train, seed=args.seed + 1
    )

    save_json(
        args.output / "samples.json",
        {
            "tinystories": tiny_samples,
            "owt": owt_samples,
        },
    )

    # (a), (b), (c)：两个 tokenizer 使用同一份 OWT 样本
    results = {
        "tiny_on_tiny": measure(
            tiny, tiny_samples, "tiny_on_tiny", args.output
        ),
        "owt_on_owt": measure(
            owt, owt_samples, "owt_on_owt", args.output
        ),
        "tiny_on_owt": measure(
            tiny, owt_samples, "tiny_on_owt", args.output
        ),
    }

    report = {
        "seed": args.seed,
        "sample_document_bodies_exclude_delimiters": True,
        "nonempty_documents_scanned": {
            "tinystories": tiny_seen,
            "owt": owt_seen,
        },
        "timing_scope": (
            "Encoding sampled documents; excludes loading, "
            "sampling, and file saving"
        ),
        "pile_bytes_assumed": PILE_BYTES,
        "results": results,
        "tiny_vs_owt_token_count_ratio_on_owt": (
            results["tiny_on_owt"]["tokens"]
            / results["owt_on_owt"]["tokens"]
        ),
    }

    save_json(args.output / "report.json", report)

    a = results["tiny_on_tiny"]
    b = results["owt_on_owt"]
    cross = results["tiny_on_owt"]

    answers = (
        "(a) On 10 sampled documents from each corpus, "
        "the TinyStories and OpenWebText tokenizers achieved "
        f"{a['bytes_per_token']:.3f} and "
        f"{b['bytes_per_token']:.3f} bytes/token, respectively.\n\n"

        "(b) On the same OpenWebText sample, the TinyStories "
        f"tokenizer produced {cross['tokens']} tokens versus "
        f"{b['tokens']} for the OpenWebText tokenizer "
        f"({cross['bytes_per_token']:.3f} versus "
        f"{b['bytes_per_token']:.3f} bytes/token).\n\n"

        "(c) Encoding throughput was "
        f"{a['bytes_per_second']/1e6:.3f} MB/s for TinyStories "
        f"and {b['bytes_per_second']/1e6:.3f} MB/s for OpenWebText. "
        "At those rates, 825 GB would take approximately "
        f"{a['pile_825gb_estimate_hours']:.2f} and "
        f"{b['pile_825gb_estimate_hours']:.2f} hours, respectively.\n\n"

        "(d) uint16 represents IDs from 0 to 65535 using "
        "two bytes per token, covering the ID ranges "
        "of the 10K and 32K vocabularies.\n"
    )

    (args.output / "answers.txt").write_text(
        answers, encoding="utf-8"
    )
    print(answers, flush=True)

    # (d)：完整数据保留原始文本及文档分隔符
    if args.serialize:
        jobs = [
            ("tinystories_train", tiny, args.tiny_train),
            ("tinystories_valid", tiny, args.tiny_valid),
            ("owt_train", owt, args.owt_train),
            ("owt_valid", owt, args.owt_valid),
        ]

        serialization = {}

        for name, tokenizer, path in jobs:
            print(f"Encoding {name}...", flush=True)

            serialization[name] = encode_to_npy(
                tokenizer,
                path,
                args.output / f"{name}.npy",
            )

            save_json(
                args.output / "serialization.json",
                serialization,
            )
            print(serialization[name], flush=True)

    print(f"Saved results to: {args.output.resolve()}")


if __name__ == "__main__":
    main()