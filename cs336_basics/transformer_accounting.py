"""CS336 transformer_accounting: arithmetic only, with no PyTorch dependency."""

from __future__ import annotations

import argparse
import json


VOCAB_SIZE = 50_257
CONFIGS = {
    "Small": (12, 768, 12),
    "Medium": (24, 1024, 16),
    "Large": (36, 1280, 20),
    "XL": (48, 1600, 25),
}


def ffn_width(d_model: int, rounding: str = "nearest") -> int:
    denominator = 3 * 64
    if rounding == "nearest":
        chunks = (8 * d_model + denominator // 2) // denominator
    elif rounding == "ceil":
        chunks = (8 * d_model + denominator - 1) // denominator
    else:
        raise ValueError("rounding must be nearest or ceil")
    return 64 * chunks


def account(
    layers: int,
    d_model: int,
    num_heads: int,
    d_ff: int,
    seq_len: int = 1024,
    vocab_size: int = VOCAB_SIZE,
) -> dict:
    L, d, h, f, S, V = layers, d_model, num_heads, d_ff, seq_len, vocab_size
    if min(L, d, h, f, S, V) <= 0 or d % h != 0:
        raise ValueError("Dimensions must be positive and d_model divisible by num_heads")

    parameter_parts = {
        "token_embeddings": V * d,
        "attention_projections": L * 4 * d * d,
        "ffn": L * 3 * d * f,
        "rmsnorm": (2 * L + 1) * d,
        "lm_head": V * d,
    }
    parameters = sum(parameter_parts.values())

    matmuls = {
        "Q/K/V projections": 6 * L * S * d * d,
        "Attention output projection": 2 * L * S * d * d,
        "QK^T, all heads": 2 * L * S * S * d,
        "Attention weights @ V, all heads": 2 * L * S * S * d,
        "FFN w1 and w3": 4 * L * S * d * f,
        "FFN w2": 2 * L * S * d * f,
        "LM head": 2 * S * d * V,
    }
    components = {
        "Attention projections": 8 * L * S * d * d,
        "Attention scores and weighted values": 4 * L * S * S * d,
        "FFN": 6 * L * S * d * f,
        "LM head": 2 * S * d * V,
    }
    flops = sum(matmuls.values())

    # Cross-check FLOPs against the number of matrix-valued weights.
    linear_weights = parameters - V * d - (2 * L + 1) * d
    assert flops == 2 * S * linear_weights + 4 * L * S * S * d
    assert flops == sum(components.values())

    return {
        "config": {
            "layers": L, "d_model": d, "num_heads": h,
            "d_ff": f, "seq_len": S, "vocab_size": V,
        },
        "parameter_parts": parameter_parts,
        "parameters": parameters,
        "fp32_parameter_bytes": 4 * parameters,
        "fp32_parameter_GB": 4 * parameters / 10**9,
        "fp32_parameter_GiB": 4 * parameters / 2**30,
        "rope_cache_bytes": 4 * L * S * (d // h),
        "matmul_flops": matmuls,
        "component_flops": components,
        "component_percent": {
            name: 100 * value / flops for name, value in components.items()
        },
        "flops": flops,
        "TFLOPs": flops / 10**12,
    }


def build_report(rounding: str = "nearest") -> dict:
    models = {
        name: account(L, d, h, ffn_width(d, rounding))
        for name, (L, d, h) in CONFIGS.items()
    }
    xl = models["XL"]
    long_xl = account(48, 1600, 25, 4288, seq_len=16_384)

    quadratic = xl["component_flops"]["Attention scores and weighted values"]
    linear = xl["flops"] - quadratic
    assert long_xl["flops"] == 16 * linear + 256 * quadratic
    assert xl["parameters"] == long_xl["parameters"] == 1_640_452_800
    assert xl["flops"] == 3_516_769_894_400

    return {
        "rounding": rounding,
        "models": models,
        "XL_long_context": long_xl,
        "long_context_flops_ratio": long_xl["flops"] / xl["flops"],
    }


def print_report(report: dict) -> None:
    xl = report["models"]["XL"]
    print("Batch size = 1; FLOPs count dense matrix multiplications using 2mnp.")
    print(f"FFN rounding: {report['rounding']}")
    print()
    print("(a)")
    print(f"Parameters: {xl['parameters']:,}")
    print(
        f"FP32 weights: {xl['fp32_parameter_bytes']:,} bytes "
        f"= {xl['fp32_parameter_GB']:.9f} GB "
        f"= {xl['fp32_parameter_GiB']:.9f} GiB"
    )
    print(f"Additional per-layer RoPE caches: {xl['rope_cache_bytes'] / 2**20:.0f} MiB")
    print()
    print("(b)")
    print("| Matrix multiplication | FLOPs | TFLOPs |")
    print("|---|---:|---:|")
    for name, value in xl["matmul_flops"].items():
        print(f"| {name} | {value:,} | {value / 10**12:.9f} |")
    print(f"Total: {xl['flops']:,} FLOPs = {xl['TFLOPs']:.9f} TFLOPs")
    print()
    print("(c)")
    for name, value in xl["component_percent"].items():
        print(f"{name}: {value:.4f}%")
    print()
    print("(d)")
    print("| Model | d_ff | TFLOPs | Projections | Scores + values | FFN | LM head |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for name, result in report["models"].items():
        p = list(result["component_percent"].values())
        print(
            f"| {name} | {result['config']['d_ff']} | {result['TFLOPs']:.9f} | "
            f"{p[0]:.4f}% | {p[1]:.4f}% | {p[2]:.4f}% | {p[3]:.4f}% |"
        )
    print()
    print("(e)")
    long_xl = report["XL_long_context"]
    print(
        f"S = 16384: {long_xl['flops']:,} FLOPs "
        f"= {long_xl['TFLOPs']:.9f} TFLOPs"
    )
    print(f"Increase: {report['long_context_flops_ratio']:.9f}x")
    for name, value in long_xl["component_percent"].items():
        print(f"{name}: {value:.4f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounding", choices=("nearest", "ceil"), default="nearest")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = build_report(args.rounding)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_report(report)


if __name__ == "__main__":
    main()
