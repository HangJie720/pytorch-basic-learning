"""state_dict round-trip for a tiny module."""
from __future__ import annotations

import torch
import torch.nn as nn


def main() -> None:
    torch.manual_seed(0)
    m = nn.Sequential(nn.Linear(4, 2), nn.ReLU(), nn.Linear(2, 1))
    sd = m.state_dict()
    m2 = nn.Sequential(nn.Linear(4, 2), nn.ReLU(), nn.Linear(2, 1))
    m2.load_state_dict(sd)
    x = torch.randn(3, 4)
    assert torch.allclose(m(x), m2(x))
    print("PASS: state_dict / load_state_dict")
    print(f"  keys={list(sd.keys())}")


if __name__ == "__main__":
    main()
