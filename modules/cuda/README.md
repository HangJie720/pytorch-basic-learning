# cuda

## What to learn

Device, current stream, caching allocator signals.

## Upstream map

| Topic | Path |
| --- | --- |
| Python API | `torch/cuda/__init__.py` |
| Streams | `c10/cuda/CUDAStream.cpp` |
| Allocator | `c10/cuda/CUDACachingAllocator.cpp` |
| Guards | `c10/cuda/CUDAGuard.h` |

## Call stack

```text
torch.cuda.Stream / tensor.cuda()
  → torch/csrc/cuda bindings
  → c10::cuda::getCurrentCUDAStream / CUDACachingAllocator
  → CUDA runtime
```

## Cases

- `case_stream_mem.py` — allocate on CUDA, print mem stats (skips if no GPU).
