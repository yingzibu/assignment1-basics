from __future__ import annotations

import math

import torch


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 1e-3):
        if lr < 0:
            raise ValueError("lr must be nonnegative")
        super().__init__(params, {"lr": lr})

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue

                state = self.state[parameter]
                t = state.get("t", 0)
                step_size = group["lr"] / math.sqrt(t + 1)
                parameter.add_(parameter.grad, alpha=-step_size)
                state["t"] = t + 1

        return loss


def run_experiment(initial_weights, lr: float, steps: int = 10):
    weights = torch.nn.Parameter(initial_weights.clone())
    optimizer = SGD([weights], lr=lr)
    losses = []

    for _ in range(steps):
        optimizer.zero_grad()
        loss = weights.square().mean()
        losses.append(loss.item())
        loss.backward()
        optimizer.step()

    final_loss = weights.detach().square().mean().item()
    return losses, final_loss


def main():
    torch.manual_seed(0)
    initial_weights = 5 * torch.randn(
        (10, 10), device="cpu", dtype=torch.float32
    )
    rates = (1.0, 10.0, 100.0, 1000.0)
    steps = 10
    results = [
        run_experiment(initial_weights, lr, steps) for lr in rates
    ]

    print("Loss before each update (10 updates total):")
    print("| t | " + " | ".join(f"lr={lr:g}" for lr in rates) + " |")
    print("|---|" + "---:|" * len(rates))
    for t in range(steps):
        values = " | ".join(f"{losses[t]:.6e}" for losses, _ in results)
        print(f"| {t} | {values} |")

    print("\nLoss after all 10 updates:")
    for lr, (losses, final_loss) in zip(rates, results):
        print(
            f"lr={lr:g}: loss={final_loss:.6e}, "
            f"final/initial={final_loss / losses[0]:.6e}"
        )


if __name__ == "__main__":
    main()
