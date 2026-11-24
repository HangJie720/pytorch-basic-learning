"""Report CUDA peer-access matrix (needed by symmetric memory)."""
from __future__ import annotations

import torch


def main() -> None:
    if not torch.cuda.is_available():
        print("SKIP: CUDA not available")
        return
    n = torch.cuda.device_count()
    print(f"devices={n}")
    for i in range(n):
        row = []
        for j in range(n):
            row.append("Y" if i == j or torch.cuda.can_device_access_peer(i, j) else ".")
        print(f"  gpu{i}: {' '.join(row)}")
    print("PASS: printed P2P matrix (Y=peer ok, .=no)")


if __name__ == "__main__":
    main()
