# pytorch-basic-learning

Source-oriented notes on how PyTorch is put together: Python APIs, bindings, ATen / c10, the Dispatcher, compile, the CUDA caching allocator, CUDA Graphs, and distributed runtimes.

Path names refer to a normal upstream tree (`torch/`, `torch/csrc/`, `aten/`, `c10/`). Each topic under [`modules/`](modules/) has an architecture diagram plus a small runnable case.

- [architecture](modules/architecture/)
- [compile](modules/compile/)
- [c10_aten](modules/c10_aten/)
- [allocator](modules/allocator/)
- [cudagraph](modules/cudagraph/)
- [distributed](modules/distributed/)
- [tensor](modules/tensor/) · [autograd](modules/autograd/) · [nn](modules/nn/) · [cuda](modules/cuda/) · [export](modules/export/)
