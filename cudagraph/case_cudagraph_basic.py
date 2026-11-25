"""Capture a tiny CUDA module with CUDAGraph and replay it."""
from __future__ import annotations

import torch
import torch.nn as nn


def main() -> None:
    if not torch.cuda.is_available():
        print("SKIP: CUDA not available")
        return

    torch.manual_seed(0)
    device = torch.device("cuda")
    model = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 4)).to(device)
    model.eval()
    static_x = torch.randn(2, 8, device=device)
    static_y = torch.empty(2, 4, device=device)

    # Warmup (allocator / cuDNN algorithms) before capture.
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            static_y.copy_(model(static_x))
    torch.cuda.current_stream().wait_stream(s)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        static_y.copy_(model(static_x))

    # Change input in-place (same storage), replay graph.
    static_x.copy_(torch.randn_like(static_x))
    g.replay()
    y_graph = static_y.clone()

    y_eager = model(static_x)
    if torch.allclose(y_graph, y_eager, rtol=1e-3, atol=1e-3):
        print("PASS: CUDAGraph replay matches eager on tiny CUDA MLP")
    else:
        max_abs = (y_graph - y_eager).abs().max().item()
        raise SystemExit(f"FAIL: max_abs={max_abs}")


if __name__ == "__main__":
    main()
