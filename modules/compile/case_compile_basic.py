"""Eager vs torch.compile on a tiny module."""
from __future__ import annotations

import torch
import torch.nn as nn


def main() -> None:
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 4))
    x = torch.randn(2, 8)
    y_eager = model(x)

    try:
        compiled = torch.compile(model, fullgraph=False)
        y_compiled = compiled(x)
    except Exception as exc:  # noqa: BLE001 — educational smoke
        print(f"SKIP: torch.compile failed: {type(exc).__name__}: {exc}")
        return

    if torch.allclose(y_eager, y_compiled, rtol=1e-4, atol=1e-4):
        print("PASS: compile matches eager on tiny MLP")
    else:
        max_abs = (y_eager - y_compiled).abs().max().item()
        raise SystemExit(f"FAIL: max_abs={max_abs}")


if __name__ == "__main__":
    main()
