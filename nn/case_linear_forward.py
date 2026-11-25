"""nn.Linear vs F.linear on the same weights."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def main() -> None:
    torch.manual_seed(0)
    m = nn.Linear(4, 3, bias=True)
    x = torch.randn(2, 4)
    y_mod = m(x)
    y_fn = F.linear(x, m.weight, m.bias)
    assert torch.allclose(y_mod, y_fn)
    print("PASS: Module forward matches F.linear")
    print(f"  params={[n for n, _ in m.named_parameters()]}")


if __name__ == "__main__":
    main()
