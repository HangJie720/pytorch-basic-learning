"""CUDA stream + memory stats smoke test."""
from __future__ import annotations

import torch


def main() -> None:
    if not torch.cuda.is_available():
        print("SKIP: CUDA not available")
        return

    device = torch.device("cuda:0")
    torch.cuda.reset_peak_memory_stats(device)
    s = torch.cuda.Stream(device=device)
    with torch.cuda.stream(s):
        x = torch.randn(1024, 1024, device=device)
        y = x @ x
        s.synchronize()
    allocated = torch.cuda.memory_allocated(device)
    peak = torch.cuda.max_memory_allocated(device)
    print("PASS: CUDA matmul on explicit stream")
    print(f"  allocated={allocated} peak={peak} y[0,0]={y[0,0].item():.4f}")


if __name__ == "__main__":
    main()
