"""CS336 AdamW resource estimates; arithmetic only."""

from __future__ import annotations

import argparse
from fractions import Fraction
import json


def number(value):
    if isinstance(value, Fraction):
        return value.numerator if value.denominator == 1 else float(value)
    return value


def account(
    vocab_size=50_257,
    context_length=1024,
    num_layers=48,
    d_model=1600,
    num_heads=25,
    d_ff=None,
    batch_size=1024,
    steps=400_000,
    memory_gb=80,
    peak_tflops=495,
    mfu=0.5,
    optimizer_ops=14,
):
    V, S, L, d, H = vocab_size, context_length, num_layers, d_model, num_heads
    f = Fraction(8 * d, 3) if d_ff is None else Fraction(d_ff)
    if min(V, S, L, d, H, f, batch_size, steps, memory_gb, peak_tflops) <= 0:
        raise ValueError("Dimensions and resource limits must be positive")
    if d % H != 0 or not 0 < mfu <= 1 or optimizer_ops not in (13, 14):
        raise ValueError("Invalid head count, MFU or optimizer FLOPs convention")

    P = 2 * V * d + L * (4 * d * d + 3 * d * f + 2 * d) + d
    assert P.denominator == 1
    P = int(P)

    # One principal saved tensor per listed operation.
    # Cross-entropy contributes one additional BSv distribution tensor.
    activation_parts = {
        "block_rmsnorms": 2 * L * S * d,
        "qkv_projections": 3 * L * S * d,
        "attention_scores_and_softmax": 2 * L * H * S * S,
        "attention_weighted_values_and_output": 2 * L * S * d,
        "ffn_w1_w3_silu_product": 4 * L * S * f,
        "ffn_w2": L * S * d,
        "final_rmsnorm": S * d,
        "logits_and_cross_entropy": 2 * S * V,
    }
    A = sum(activation_parts.values())
    assert A == S * (L * (8 * d + 4 * f + 2 * H * S) + d + 2 * V)

    activation_bytes = 4 * A
    static_bytes = 16 * P
    budget = Fraction(str(memory_gb)) * 10**9
    max_batch = max(0, int((budget - static_bytes) // activation_bytes))

    forward_flops = (
        L * (8 * S * d * d + 4 * S * S * d + 6 * S * d * f)
        + 2 * S * d * V
    )
    assert forward_flops == (
        2 * S * (P - V * d - (2 * L + 1) * d)
        + 4 * L * S * S * d
    )
    optimizer_flops = optimizer_ops * P
    step_flops = 3 * batch_size * forward_flops + optimizer_flops
    effective_rate = (
        Fraction(str(peak_tflops)) * Fraction(str(mfu)) * 10**12
    )
    hours = steps * step_flops / effective_rate / 3600

    return {
        "assumptions": {
            "dtype": "float32",
            "activations": "one principal output per listed operation; CE adds one BSv tensor",
            "input_and_output_embeddings": "independent weights",
            "attention": "dense scores and probabilities",
            "memory_GB_unit": 10**9,
        },
        "config": {
            "vocab_size": V, "context_length": S, "num_layers": L,
            "d_model": d, "num_heads": H, "d_ff": str(f),
            "effective_batch_size": batch_size, "steps": steps,
            "peak_tflops": peak_tflops, "mfu": mfu,
        },
        "parameters": P,
        "parameter_bytes": 4 * P,
        "gradient_bytes": 4 * P,
        "optimizer_state_bytes": 8 * P,
        "activation_elements_per_batch_item": number(A),
        "activation_parts_per_batch_item": {
            k: number(v) for k, v in activation_parts.items()
        },
        "activation_bytes_per_batch_item": number(activation_bytes),
        "static_bytes": static_bytes,
        "memory_GB_slope": float(activation_bytes / 10**9),
        "memory_GB_intercept": static_bytes / 10**9,
        "max_batch_size": max_batch,
        "memory_at_max_batch_GB": float(
            (static_bytes + max_batch * activation_bytes) / 10**9
        ),
        "memory_at_next_batch_GB": float(
            (static_bytes + (max_batch + 1) * activation_bytes) / 10**9
        ),
        "optimizer_ops_per_parameter": optimizer_ops,
        "optimizer_step_flops": optimizer_flops,
        "forward_flops_per_sequence": number(forward_flops),
        "training_step_flops": number(step_flops),
        "effective_flops_per_second": number(effective_rate),
        "training_hours": float(hours),
        "training_days": float(hours / 24),
    }


def print_report(result):
    r = result
    print("Simplified retained-activation estimate, all floating tensors FP32.")
    print(f"d_ff = {r['config']['d_ff']}")
    print()
    print("(a)")
    print(f"Parameters: {r['parameters']:,}")
    print(f"Parameter memory: {r['parameter_bytes']:,} bytes")
    print(f"Gradient memory: {r['gradient_bytes']:,} bytes")
    print(f"AdamW state memory: {r['optimizer_state_bytes']:,} bytes")
    print(
        "Activation memory: "
        f"{r['activation_bytes_per_batch_item']:,} * batch_size bytes"
    )
    print()
    print("(b)")
    print(
        f"M(B) = {r['memory_GB_slope']:.9f} * B "
        f"+ {r['memory_GB_intercept']:.9f} GB"
    )
    print(f"Maximum theoretical batch size: {r['max_batch_size']}")
    print(f"Memory at maximum: {r['memory_at_max_batch_GB']:.9f} GB")
    print(f"Memory at next batch: {r['memory_at_next_batch_GB']:.9f} GB")
    print()
    print("(c)")
    print(
        f"AdamW: {r['optimizer_ops_per_parameter']} * P "
        f"= {r['optimizer_step_flops']:,} FLOPs"
    )
    print()
    print("(d)")
    print(f"Forward FLOPs per sequence: {r['forward_flops_per_sequence']:,}")
    print(f"Training FLOPs per effective batch: {r['training_step_flops']:,}")
    print(f"Effective compute rate: {r['effective_flops_per_second']:,} FLOP/s")
    print(f"Training time: {r['training_hours']:.6f} hours")
    print(f"Training time: {r['training_days']:.6f} days")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d-ff", type=int, default=None)
    parser.add_argument("--optimizer-ops", type=int, choices=(13, 14), default=14)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=400_000)
    parser.add_argument("--memory-gb", type=float, default=80)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = account(
        d_ff=args.d_ff,
        optimizer_ops=args.optimizer_ops,
        batch_size=args.batch_size,
        steps=args.steps,
        memory_gb=args.memory_gb,
    )
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print_report(result)


if __name__ == "__main__":
    main()
