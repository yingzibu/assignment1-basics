"""Configurable CS336 language-model training entry point.

Run from the repository root:
    python -m cs336_basics.train --help
    python -m cs336_basics.train --train-data train.npy --val-data valid.npy --check-data
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np


DEFAULTS = {
    "vocab_size": 10000,
    "context_length": 256,
    "d_model": 256,
    "num_layers": 4,
    "num_heads": 8,
    "d_ff": None,
    "rope_theta": 10000.0,
    "rms_eps": 1e-5,
    "batch_size": 16,
    "grad_accum_steps": 1,
    "max_steps": 5000,
    "max_lr": 3e-4,
    "min_lr": 3e-5,
    "warmup_iters": 100,
    "cosine_cycle_iters": None,
    "beta1": 0.9,
    "beta2": 0.999,
    "adam_eps": 1e-8,
    "weight_decay": 0.1,
    "max_l2_norm": 1.0,
    "log_every": 10,
    "eval_every": 100,
    "eval_batches": 20,
    "save_every": 500,
    "seed": 42,
}
MODEL_KEYS = (
    "vocab_size", "context_length", "d_model", "num_layers",
    "num_heads", "d_ff", "rope_theta", "rms_eps",
)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", help="Training tokens in a 1D integer .npy file")
    parser.add_argument("--val-data", help="Validation tokens in a 1D integer .npy file")
    parser.add_argument("--out-dir", help="Output directory; default: artifacts/training")
    parser.add_argument("--device", help="auto, cpu, cuda, cuda:0, or mps")
    parser.add_argument("--resume", help="Checkpoint path to restore")
    parser.add_argument(
        "--check-data", action="store_true",
        help="Scan token ranges and validate configuration, then exit without training",
    )
    for name, default in DEFAULTS.items():
        value_type = float if isinstance(default, float) else int
        parser.add_argument(
            "--" + name.replace("_", "-"), type=value_type, default=None,
            help=f"Default: {default}; resume uses the saved value unless overridden",
        )
    return parser


def resolve_config(args, checkpoint=None):
    saved = checkpoint.get("config", {}) if checkpoint is not None else {}
    config = DEFAULTS.copy()
    config.update({key: saved[key] for key in DEFAULTS if key in saved})
    for key in DEFAULTS:
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    config["train_data"] = args.train_data or saved.get("train_data")
    config["val_data"] = args.val_data or saved.get("val_data")
    config["out_dir"] = args.out_dir or (
        str(Path(args.resume).parent) if args.resume else "artifacts/training"
    )
    config["device"] = args.device or "auto"
    if not config["train_data"] or not config["val_data"]:
        raise ValueError("--train-data and --val-data are required for a new run")
    if config["d_ff"] is None:
        config["d_ff"] = 64 * math.ceil(8 * config["d_model"] / (3 * 64))
    if config["cosine_cycle_iters"] is None:
        config["cosine_cycle_iters"] = config["max_steps"]
    for key in MODEL_KEYS:
        if key in saved and config[key] != saved[key]:
            raise ValueError(f"Cannot change model setting {key} when resuming")
    positive = (
        "vocab_size", "context_length", "d_model", "num_heads", "d_ff",
        "batch_size", "grad_accum_steps", "max_steps", "log_every",
        "eval_every", "eval_batches", "save_every",
    )
    for key in positive:
        if config[key] <= 0:
            raise ValueError(f"{key} must be positive")
    if config["num_layers"] < 0:
        raise ValueError("num_layers must be nonnegative")
    if config["d_model"] % config["num_heads"]:
        raise ValueError("d_model must be divisible by num_heads")
    if (config["d_model"] // config["num_heads"]) % 2:
        raise ValueError("RoPE requires an even dimension per attention head")
    if not 0 <= config["min_lr"] <= config["max_lr"]:
        raise ValueError("Require 0 <= min_lr <= max_lr")
    if not 0 <= config["warmup_iters"] < config["cosine_cycle_iters"]:
        raise ValueError("Require 0 <= warmup_iters < cosine_cycle_iters")
    for key in ("beta1", "beta2"):
        if not 0 <= config[key] < 1:
            raise ValueError(f"{key} must be in [0, 1)")
    for key in ("adam_eps", "weight_decay", "max_l2_norm"):
        if not config[key] >= 0:
            raise ValueError(f"{key} must be nonnegative")
    for key in ("rms_eps", "rope_theta"):
        if not config[key] > 0:
            raise ValueError(f"{key} must be positive")
    if not 0 <= config["seed"] < 2**32:
        raise ValueError("seed must be in [0, 2**32)")
    for key, value in config.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"{key} must be finite")
    return config


def load_tokens(path, context_length, vocab_size, full_scan=False):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Token file not found: {path}")
    data = np.load(path, mmap_mode="r", allow_pickle=False)
    if not isinstance(data, np.ndarray) or data.ndim != 1:
        raise ValueError(f"{path}: expected a 1D .npy array")
    if not np.issubdtype(data.dtype, np.integer):
        raise ValueError(f"{path}: token IDs must have an integer dtype")
    if len(data) <= context_length:
        raise ValueError(f"{path}: need at least context_length + 1 tokens")
    limit = len(data) if full_scan else min(len(data), 1_000_000)
    minimum = maximum = int(data[0])
    for start in range(0, limit, 1_000_000):
        chunk = data[start:min(start + 1_000_000, limit)]
        minimum = min(minimum, int(chunk.min()))
        maximum = max(maximum, int(chunk.max()))
    if minimum < 0 or maximum >= vocab_size:
        raise ValueError(
            f"{path}: token range [{minimum}, {maximum}] "
            f"is outside [0, {vocab_size - 1}]"
        )
    info = {
        "path": str(path), "tokens": len(data), "dtype": str(data.dtype),
        "checked_tokens": limit, "min_checked_id": minimum, "max_checked_id": maximum,
    }
    return data, info


def run_training(config, train_data, val_data, checkpoint=None):
    import torch

    from .adamw import AdamW
    from .cross_entropy import cross_entropy
    from .data_loading import get_batch
    from .gradient_clipping import gradient_clipping
    from .learning_rate_schedule import get_lr_cosine_schedule
    from .transformer_lm import TransformerLM

    if config["device"] == "auto":
        if torch.cuda.is_available():
            config["device"] = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            config["device"] = "mps"
        else:
            config["device"] = "cpu"
    device = torch.device(config["device"])
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    model_args = {key: config[key] for key in MODEL_KEYS if key != "rms_eps"}
    model = TransformerLM(**model_args, eps=config["rms_eps"], device=device, dtype=torch.float32)
    optimizer = AdamW(
        model.parameters(), lr=config["max_lr"],
        betas=(config["beta1"], config["beta2"]),
        eps=config["adam_eps"], weight_decay=config["weight_decay"],
    )
    resuming = checkpoint is not None
    completed_steps = 0
    best_val_loss = None
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        completed_steps = int(checkpoint["iteration"])
        best_val_loss = checkpoint.get("best_val_loss")
        if "numpy_rng_state" in checkpoint:
            state = checkpoint["numpy_rng_state"]
            np.random.set_state((state[0], np.asarray(state[1], dtype=np.uint32), *state[2:]))
        if "python_rng_state" in checkpoint:
            random.setstate(checkpoint["python_rng_state"])
        if "torch_rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["torch_rng_state"])
        if device.type == "cuda" and "cuda_rng_state" in checkpoint:
            torch.cuda.set_rng_state(checkpoint["cuda_rng_state"], device)
        if device.type == "mps" and "mps_rng_state" in checkpoint:
            torch.mps.set_rng_state(checkpoint["mps_rng_state"])
        checkpoint.clear()
    # Apply explicit optimizer overrides after restoring its state.
    for group in optimizer.param_groups:
        group["betas"] = (config["beta1"], config["beta2"])
        group["eps"] = config["adam_eps"]
        group["weight_decay"] = config["weight_decay"]
    checkpoint = None  # Release CPU checkpoint tensors after restoration.
    out_dir = Path(config["out_dir"]).expanduser()
    if (out_dir / "latest.pt").exists() and not resuming:
        raise FileExistsError(f"{out_dir}/latest.pt exists; use --resume or a new --out-dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(
        f"device={device} parameters={sum(p.numel() for p in model.parameters()):,} "
        f"effective_batch_size={config['batch_size'] * config['grad_accum_steps']} "
        f"completed_steps={completed_steps}",
        flush=True,
    )

    def evaluate():
        training_mode = model.training
        numpy_state = np.random.get_state()
        np.random.seed(config["seed"] + 1 if config["seed"] < 2**32 - 1 else 0)
        model.eval()
        try:
            total = 0.0
            with torch.no_grad():
                for _ in range(config["eval_batches"]):
                    x, y = get_batch(
                        val_data, config["batch_size"], config["context_length"], str(device)
                    )
                    total += cross_entropy(model(x), y).item()
            value = total / config["eval_batches"]
            if not math.isfinite(value):
                raise RuntimeError("Validation loss is non-finite")
            return value
        finally:
            np.random.set_state(numpy_state)
            model.train(training_mode)

    def save(name):
        numpy_state = np.random.get_state()
        state = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "iteration": completed_steps,
            "config": config.copy(),
            "best_val_loss": best_val_loss,
            "numpy_rng_state": (
                numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]
            ),
            "python_rng_state": random.getstate(),
            "torch_rng_state": torch.get_rng_state(),
        }
        if device.type == "cuda":
            state["cuda_rng_state"] = torch.cuda.get_rng_state(device)
        if device.type == "mps" and hasattr(torch.mps, "get_rng_state"):
            state["mps_rng_state"] = torch.mps.get_rng_state()
        destination = out_dir / name
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        torch.save(state, temporary)
        temporary.replace(destination)

    model.train()
    last_log_step = completed_steps
    log_started = time.perf_counter()
    loss_sum = 0.0
    with (out_dir / "metrics.jsonl").open("a", encoding="utf-8") as log:
        def emit(record):
            log.write(json.dumps(record) + "\n")
            log.flush()
            print(json.dumps(record), flush=True)

        for it in range(completed_steps, config["max_steps"]):
            lr = get_lr_cosine_schedule(
                it, config["max_lr"], config["min_lr"],
                config["warmup_iters"], config["cosine_cycle_iters"],
            )
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            step_loss = torch.zeros((), device=device)
            for _ in range(config["grad_accum_steps"]):
                x, y = get_batch(
                    train_data, config["batch_size"], config["context_length"], str(device)
                )
                loss = cross_entropy(model(x), y) / config["grad_accum_steps"]
                loss.backward()
                step_loss += loss.detach()
            if not torch.isfinite(step_loss):
                raise RuntimeError(f"Non-finite loss before update {it + 1}; inspect the learning rate")
            gradient_clipping(model.parameters(), config["max_l2_norm"])
            optimizer.step()
            completed_steps = it + 1
            loss_sum += step_loss.item()
            final_step = completed_steps == config["max_steps"]
            if completed_steps % config["log_every"] == 0 or final_step:
                updates = completed_steps - last_log_step
                elapsed = time.perf_counter() - log_started
                tokens = updates * config["batch_size"] * config["grad_accum_steps"] * config["context_length"]
                emit({
                    "split": "train", "step": completed_steps,
                    "loss": loss_sum / updates, "lr": lr,
                    "tokens_per_second": tokens / max(elapsed, 1e-9),
                })
                loss_sum = 0.0
                last_log_step = completed_steps
                log_started = time.perf_counter()
            if completed_steps % config["eval_every"] == 0 or final_step:
                val_loss = evaluate()
                emit({"split": "validation", "step": completed_steps, "loss": val_loss})
                if best_val_loss is None or val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save("best.pt")
            if completed_steps % config["save_every"] == 0 or final_step:
                save("latest.pt")
    print(f"Completed {completed_steps} updates. Checkpoints: {out_dir.resolve()}", flush=True)


def main():
    parser = build_parser()
    args = parser.parse_args()
    checkpoint = None
    if args.resume:
        import torch

        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
    try:
        config = resolve_config(args, checkpoint)
        train_data, train_info = load_tokens(
            config["train_data"], config["context_length"], config["vocab_size"], args.check_data
        )
        val_data, val_info = load_tokens(
            config["val_data"], config["context_length"], config["vocab_size"], args.check_data
        )
        config["train_data"] = train_info["path"]
        config["val_data"] = val_info["path"]
        if checkpoint is not None:
            completed = int(checkpoint["iteration"])
            if completed < 0 or completed > config["max_steps"]:
                raise ValueError("--max-steps must be at least the saved iteration count")
    except (ValueError, TypeError, FileNotFoundError) as error:
        parser.error(str(error))
    print(json.dumps({"train": train_info, "validation": val_info, "config": config}, indent=2))
    if args.check_data:
        print("Data checks passed; no training executed.")
        return
    if checkpoint is not None and int(checkpoint["iteration"]) == config["max_steps"]:
        print("The checkpoint already contains max_steps completed updates.")
        return
    run_training(config, train_data, val_data, checkpoint)


if __name__ == "__main__":
    main()

