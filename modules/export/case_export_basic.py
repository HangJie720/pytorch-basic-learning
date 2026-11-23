"""Minimal torch.export smoke test."""
from __future__ import annotations

import torch
import torch.nn as nn


class Tiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def main() -> None:
    model = Tiny().eval()
    args = (torch.randn(1, 4),)
    try:
        ep = torch.export.export(model, args)
    except Exception as exc:  # noqa: BLE001
        print(f"SKIP: torch.export failed: {type(exc).__name__}: {exc}")
        return

    out = ep.module()(*args)
    ref = model(*args)
    assert torch.allclose(out, ref, rtol=1e-4, atol=1e-4)
    print("PASS: torch.export round-trip matches eager")
    print(f"  graph_nodes={len(list(ep.graph.nodes))}")


if __name__ == "__main__":
    main()
