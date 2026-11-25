# pytorch-basic-learning

Source-oriented notes on how PyTorch is put together: Python APIs, bindings,
ATen / c10, the Dispatcher, compile, the CUDA caching allocator, CUDA Graphs,
and distributed runtimes.

Path names refer to a normal upstream tree (`torch/`, `torch/csrc/`, `aten/`,
`c10/`). Each topic directory has:

- a short **role / design** write-up (with links to official PyTorch blogs/docs)
- an **architecture diagram**
- a small **runnable case** where useful

## Topics

- [architecture](architecture/) — layered source map
- [compile](compile/) — `torch.compile` / Dynamo / Inductor
- [c10_aten](c10_aten/) — Dispatcher & DispatchKey
- [allocator](allocator/) — CUDACachingAllocator & expandable segments
- [cudagraph](cudagraph/) — CUDA Graph capture / replay
- [distributed](distributed/) — c10d / NCCL / FSDP / …
- [tensor](tensor/) · [autograd](autograd/) · [nn](nn/) · [cuda](cuda/) · [export](export/)
