# compile (`torch.compile`)

## Role

`torch.compile` is the **opt-in compiler API** introduced as the centerpiece of
PyTorch 2.x. It wraps a function or `nn.Module` and returns a compiled callable that
aims to run faster while staying **additive**: if compile fails or graph-breaks, you
still get correct eager behavior.

Under the hood (from the [PyTorch 2.0 blog](https://pytorch.org/blog/pytorch-2.0-release/)):

| Piece | Job |
| --- | --- |
| **TorchDynamo** | Safe graph capture via Python frame evaluation hooks |
| **AOTAutograd** | Trace forward + backward ahead of time |
| **PrimTorch** | Canonicalize ~2000 ops down to a smaller primitive set |
| **TorchInductor** | Codegen (Triton for NVIDIA/AMD GPUs; C++/vector for CPU) |

## How a call flows

1. First (or shape-changing) call: Dynamo intercepts bytecode, builds an FX graph with
   guards (tensor metadata / control-flow assumptions).
2. FX passes clean / specialize the graph.
3. Inductor schedules and emits kernels (or fused Triton kernels).
4. On later calls with matching guards: run the compiled artifact.
5. On guard failure or unsupported ops: **graph break** or full fallback into the
   ATen Dispatcher — same path eager would take.

Modes such as `reduce-overhead` / `max-autotune` additionally lean on CUDA Graphs
(see [`../cudagraph`](../cudagraph/)) when the region is static enough.

<img src="images/torch_compile_flow.jpg" width="640" alt="torch.compile path" />

## What this module teaches

- Where compile sits relative to eager (side path, not a replacement runtime).
- Why “compile failed / graph break” is expected for dynamic Python.
- How Inductor output still shares Device / Allocator / Stream with the rest of PyTorch.

## Upstream map

| Topic | Path |
| --- | --- |
| Entry | `torch/_dynamo/eval_frame.py`, `torch/__init__.py` (`compile`) |
| FX | `torch/fx/` |
| Inductor | `torch/_inductor/` |
| C++ helpers | `torch/csrc/dynamo/`, `torch/csrc/inductor/` |

## Cases

- `case_compile_basic.py` — eager vs compile on a tiny MLP (skips if compile unavailable).

```bash
cd modules/compile
python case_compile_basic.py
```

## References

- [PyTorch 2.0: Our next generation release](https://pytorch.org/blog/pytorch-2.0-release/)
- [Get started with PyTorch 2.0 / torch.compile](https://pytorch.org/get-started/pytorch-2.0/)
- [Accelerating Generative AI with PyTorch (compile + CUDA Graph)](https://pytorch.org/blog/accelerating-generative-ai-4/)
