# export

## What to learn

Graph capture via `torch.export` / FX, vs TorchScript (`torch.jit`).

## Upstream map

| Topic | Path |
| --- | --- |
| Export Python | `torch/export/` |
| FX | `torch/fx/` |
| JIT | `torch/jit/`, `torch/csrc/jit/` |

## Call stack (export sketch)

```text
torch.export.export(model, args)
  → dynamo/export tracing + soundness checks
  → ExportedProgram (FX graph + state)
```

## Cases

- `case_export_basic.py` — export a tiny module when available.
