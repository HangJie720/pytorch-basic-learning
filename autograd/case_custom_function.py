"""Minimal custom autograd Function."""
from __future__ import annotations

import torch


class MulByTwo(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        return x * 2

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        return grad_out * 2


def main() -> None:
    x = torch.tensor([1.5, -2.0], requires_grad=True)
    y = MulByTwo.apply(x)
    y.sum().backward()
    assert torch.allclose(x.grad, torch.tensor([2.0, 2.0]))
    print("PASS: custom Function forward/backward")
    print(f"  y.grad_fn={y.grad_fn}")


if __name__ == "__main__":
    main()
