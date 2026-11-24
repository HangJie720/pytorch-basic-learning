"""Inspect grad_fn chain for a tiny graph."""
from __future__ import annotations

import torch


def main() -> None:
    x = torch.tensor([2.0], requires_grad=True)
    y = (x * 3).exp()
    y.backward()
    assert x.grad is not None
    print("PASS: grad_fn chain")
    print(f"  y.grad_fn={y.grad_fn}")
    print(f"  next={y.grad_fn.next_functions}")
    print(f"  x.grad={x.grad.item()}")


if __name__ == "__main__":
    main()
