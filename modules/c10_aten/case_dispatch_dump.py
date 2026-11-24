"""Dump a slice of the Dispatcher table for aten::add.Tensor."""
from __future__ import annotations

import torch


def main() -> None:
    dump = torch._C._dispatch_dump("aten::add.Tensor")
    print("PASS: dispatcher dump for aten::add.Tensor")
    print(dump[:1500])
    if len(dump) > 1500:
        print("... [truncated]")


if __name__ == "__main__":
    main()
