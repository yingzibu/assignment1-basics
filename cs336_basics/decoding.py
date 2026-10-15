"""Autoregressive decoding with temperature and nucleus sampling."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

EOS_TOKEN = "<|endoftext|>"


def _check_sampling(temperature: float, top_p: float) -> None:
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError("temperature must be finite and nonnegative")
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> torch.Tensor:
    """Sample from 1D vocabulary logits; return a LongTensor of shape (1,)."""
    import torch

    from .softmax import softmax

    _check_sampling(temperature, top_p)
    if logits.ndim != 1 or logits.numel() == 0:
        raise ValueError("logits must be a nonempty 1D vocabulary tensor")
    with torch.no_grad():
        dtype = torch.float64 if logits.dtype == torch.float64 else torch.float32
        scores = logits.to(dtype=dtype)
        maximum = scores.max()
        if not torch.isfinite(maximum):
            raise ValueError("The largest logit must be finite")
        if temperature == 0:
            return scores.argmax(dim=-1, keepdim=True)

        # Shifting before division avoids positive overflow at small temperatures.
        probabilities = softmax((scores - maximum) / temperature, dim=-1)
        if top_p == 1:
            return torch.multinomial(probabilities, num_samples=1)

        sorted_probs, sorted_ids = torch.sort(probabilities, descending=True)
        reached = sorted_probs.cumsum(dim=-1) >= top_p
        remove = torch.zeros_like(reached)
        # Keep the token that first reaches or crosses the threshold.
        remove[1:] = reached[:-1]
        sorted_probs = sorted_probs.masked_fill(remove, 0.0)
        sorted_probs = sorted_probs / sorted_probs.sum()
        picked = torch.multinomial(sorted_probs, num_samples=1)
        return sorted_ids[picked]


def generate(
    model,
    prompt_ids: list[int] | torch.Tensor,
    max_new_tokens: int = 128,
    temperature: float = 0.8,
    top_p: float = 0.9,
    eos_token_id: int | None = None,
    context_length: int | None = None,
) -> list[int]:
    """Return prompt IDs plus generated IDs, including a generated EOS token."""
    import torch

    _check_sampling(temperature, top_p)
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be nonnegative")
    capacity = getattr(model, "context_length", None)
    if context_length is None:
        context_length = capacity
    if context_length is None or context_length <= 0:
        raise ValueError("Provide a positive context_length or model.context_length")
    if capacity is not None and context_length > capacity:
        raise ValueError("context_length exceeds the model's configured capacity")

    parameter = next(model.parameters(), None)
    device = parameter.device if parameter is not None else torch.device("cpu")
    ids = torch.as_tensor(prompt_ids, dtype=torch.long, device=device)
    if ids.ndim != 1 or ids.numel() == 0:
        raise ValueError("prompt_ids must be a nonempty 1D sequence")
    vocab_size = getattr(model, "vocab_size", None)
    if (ids < 0).any() or (vocab_size is not None and (ids >= vocab_size).any()):
        raise ValueError("Prompt token IDs are outside the model vocabulary")
    if eos_token_id is not None and (
        eos_token_id < 0 or (vocab_size is not None and eos_token_id >= vocab_size)
    ):
        raise ValueError("eos_token_id is outside the model vocabulary")

    tokens = ids.unsqueeze(0)
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # The model assigns positions 0..window_length-1 to this window.
                context = tokens[:, -context_length:]
                logits = model(context)[0, -1, :]
                next_token = sample_next_token(logits, temperature, top_p)
                tokens = torch.cat((tokens, next_token.reshape(1, 1)), dim=-1)
                if eos_token_id is not None and next_token.item() == eos_token_id:
                    break
        return tokens[0].tolist()
    finally:
        model.train(was_training)


def decode(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.8,
    top_p: float = 0.9,
    include_prompt: bool = True,
    eos_token: str = EOS_TOKEN,
) -> str:
    """Generate text; omit generated EOS and the bootstrap used for an empty prompt."""
    eos_ids = tokenizer.encode(eos_token)
    if len(eos_ids) != 1:
        raise ValueError("Register eos_token as a single special token in the tokenizer")
    eos_id = eos_ids[0]
    prompt_ids = tokenizer.encode(prompt)
    conditioning_ids = prompt_ids if prompt_ids else [eos_id]
    output_ids = generate(
        model, conditioning_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        eos_token_id=eos_id,
    )
    completion_ids = output_ids[len(conditioning_ids):]
    if completion_ids and completion_ids[-1] == eos_id:
        completion_ids = completion_ids[:-1]
    text_ids = prompt_ids + completion_ids if include_prompt else completion_ids
    return tokenizer.decode(text_ids)


def _model_config(checkpoint: dict, config_path: Path | None = None) -> dict:
    if config_path is not None:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    else:
        config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise ValueError("Checkpoint needs training config; supply --config config.json")
    required = ("vocab_size", "context_length", "d_model", "num_layers", "num_heads")
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError("Missing model settings: " + ", ".join(missing))
    result = {key: config[key] for key in required}
    d_ff = config.get("d_ff")
    result["d_ff"] = d_ff if d_ff is not None else 64 * math.ceil(8 * config["d_model"] / (3 * 64))
    result["rope_theta"] = config.get("rope_theta", 10000.0)
    result["eps"] = config.get("rms_eps", config.get("eps", 1e-5))
    return result


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vocab", type=Path, required=True, help="BPE vocab.pkl")
    parser.add_argument("--merges", type=Path, required=True, help="BPE merges.pkl")
    parser.add_argument("--config", type=Path, help="Model config JSON for older checkpoints")
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8, help="0 selects greedy decoding")
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--completion-only", action="store_true")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        _check_sampling(args.temperature, args.top_p)
        if args.max_new_tokens < 0:
            raise ValueError("max_new_tokens must be nonnegative")
        for label, path in (
            ("checkpoint", args.checkpoint), ("vocab", args.vocab),
            ("merges", args.merges), ("config", args.config),
        ):
            if path is not None and not path.is_file():
                raise FileNotFoundError(f"{label} file not found: {path.resolve()}")
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))

    import torch

    from .tokenizer import Tokenizer
    from .transformer_lm import TransformerLM

    if args.device == "auto":
        if torch.cuda.is_available():
            args.device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config_path = args.config
    if config_path is None and not checkpoint.get("config"):
        candidate = args.checkpoint.parent / "config.json"
        if candidate.is_file():
            config_path = candidate
    try:
        model_config = _model_config(checkpoint, config_path)
        weights = checkpoint["model_state_dict"]
    except (ValueError, KeyError) as error:
        parser.error(str(error))
    del checkpoint
    model = TransformerLM(**model_config, device=device, dtype=torch.float32)
    model.load_state_dict(weights)
    del weights
    tokenizer = Tokenizer.from_files(args.vocab, args.merges, special_tokens=[EOS_TOKEN])
    if set(tokenizer.vocab) != set(range(model_config["vocab_size"])):
        parser.error("Tokenizer vocabulary IDs must match the checkpoint vocabulary")
    torch.manual_seed(args.seed)
    print(decode(
        model, tokenizer, args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        include_prompt=not args.completion_only,
    ))


if __name__ == "__main__":
    main()

