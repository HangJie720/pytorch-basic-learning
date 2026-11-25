# cudagraph (`torch.cuda.CUDAGraph`)

## Role

**CUDA Graphs** (CUDA 10+) package a sequence of GPU operations into one
executable graph. Instead of the CPU launching every kernel separately, a
replay submits the whole graph with a single `cudaGraphLaunch`, cutting Python /
C++ / driver launch overhead. PyTorch exposes this via stream capture
(`torch.cuda.CUDAGraph`, `torch.cuda.graph`, `make_graphed_callables`) and can
also emit cudagraphs from **Inductor** under `torch.compile` when shapes and
control flow are static enough.

From the official
[Accelerating PyTorch with CUDA Graphs](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/)
blog:

- Work issued to a **capturing stream** is recorded, not executed for real.
- Each **replay** runs the same kernels with the **same arguments / addresses**.
- You refresh inputs by writing new data into those fixed buffers between
  replays.
- Main win: elide CPU overhead when the GPU region is otherwise CPU-bound.
- NCCL collectives can participate in graphs so multi-GPU launches stay
  tightly clustered (less jitter across ranks).

## How capture works in PyTorch

1. Warm up once (allocator / cuDNN algo selection) **outside** capture.
2. `CUDAGraph.capture_begin` / `torch.cuda.graph(...)` puts the stream in
   capture mode.
3. Run the static region (same shapes, no unexpected CPU sync / data-dependent
   control flow).
4. `capture_end` builds a `cudaGraphExec_t`.
5. Later: fill static inputs → `graph.replay()`.

Constraints (graph-safety): fixed tensor addresses and shapes, limited host
sync, careful use of RNG / CPU ops. The caching allocator uses a **private pool**
during capture so addresses remain valid on replay (see [`../allocator`](../allocator/)).

<img src="images/cuda_graph_architecture.jpg" width="640" alt="CUDA Graph architecture" />

## Relation to `torch.compile`

- `mode="reduce-overhead"` / `max-autotune` may wrap compiled regions in
  cudagraphs automatically.
- Inductor’s **cudagraph trees** handle a family of related graphs and allocator
  checkpoints between captures.
- Regions that mutate inputs in ways that break pointer stability may need
  `max-autotune-no-cudagraphs` or explicit graph-safe coding patterns (see
  generative-AI acceleration blogs).

## Upstream map

| Topic | Path |
| --- | --- |
| Python API | `torch/cuda/graphs.py`, `torch.cuda.graph`, `make_graphed_callables` |
| C++ / bindings | `torch/csrc/cuda/Graph.cpp`, `CUDAGraph.h` |
| Inductor | `torch/_inductor/cudagraph_trees.py`, cudagraph partitioning |

## Cases

- `case_cudagraph_basic.py` — capture a tiny CUDA MLP once, replay, compare to eager (skips if no CUDA).

```bash
cd modules/cudagraph
python case_cudagraph_basic.py
```

## References

- [Accelerating PyTorch with CUDA Graphs](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/)
- [Accelerating Generative AI with PyTorch IV](https://pytorch.org/blog/accelerating-generative-ai-4/) — compile + CUDA Graph constraints
- [CUDA semantics](https://pytorch.org/docs/stable/notes/cuda.html) — graphs + allocator interaction
