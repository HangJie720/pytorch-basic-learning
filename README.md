# pytorch-basic-learning

Source-oriented notes on how PyTorch is put together: Python APIs, bindings,
ATen / c10, the Dispatcher, compile, the CUDA caching allocator, CUDA Graphs,
and distributed runtimes.

Path names refer to a normal upstream tree (`torch/`, `torch/csrc/`, `aten/`,
`c10/`). Each topic under [`modules/`](modules/) has:

- a short **role / design** write-up (with links to official PyTorch blogs/docs)
- an **architecture diagram**
- a small **runnable case** where useful

## Modules

- [architecture](modules/architecture/) — layered source map
- [compile](modules/compile/) — `torch.compile` / Dynamo / Inductor
- [c10_aten](modules/c10_aten/) — Dispatcher & DispatchKey
- [allocator](modules/allocator/) — CUDACachingAllocator & expandable segments
- [cudagraph](modules/cudagraph/) — CUDA Graph capture / replay
- [distributed](modules/distributed/) — c10d / NCCL / FSDP / …
- [tensor](modules/tensor/) · [autograd](modules/autograd/) · [nn](modules/nn/) · [cuda](modules/cuda/) · [export](modules/export/)
