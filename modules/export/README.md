# export (`torch.export`)

## Role

`torch.export` produces an **ExportedProgram**: a sound FX graph plus state,
intended for deployment / further compilers. Unlike eager or a loosely captured
`torch.compile` region, export aims for a **full-graph, guard-checked** artifact
you can serialize and lower. It shares Dynamo / FX infrastructure with compile
but emphasizes soundness and a stable program representation over peak
eager-side speedups.

Contrast with TorchScript (`torch.jit`): export is the PyTorch 2.x direction for
graph capture going forward.

## How export flows

```text
torch.export.export(model, args)
  → Dynamo / export tracing + soundness checks
  → ExportedProgram (FX graph + state_dict-like state)
```

## Upstream map

| Topic | Path |
| --- | --- |
| Export Python | `torch/export/` |
| FX | `torch/fx/` |
| JIT (legacy) | `torch/jit/`, `torch/csrc/jit/` |

## Cases

- `case_export_basic.py` — export a tiny module when available.

```bash
cd modules/export
python case_export_basic.py
```

## References

- [torch.export docs](https://pytorch.org/docs/stable/export.html)
- [PyTorch 2.0 release blog](https://pytorch.org/blog/pytorch-2.0-release/)
