# c10 + ATen reading guide

No runtime cases here — this folder indexes **source landing spots** while you study other modules.

## c10

| Concern | Files |
| --- | --- |
| Tensor / Storage | `c10/core/TensorImpl.*`, `Storage*` |
| Device / Stream concepts | `c10/core/Device.*`, `c10/cuda/CUDAStream.*` |
| Allocator | `c10/cuda/CUDACachingAllocator.*` |
| DispatchKey | `c10/core/DispatchKey.h`, `DispatchKeySet.*` |

## ATen

| Concern | Files |
| --- | --- |
| Dispatcher | `aten/src/ATen/core/dispatch/Dispatcher.*` |
| Native CPU | `aten/src/ATen/native/` |
| CUDA kernels | `aten/src/ATen/native/cuda/` |
| BLAS | `aten/src/ATen/cuda/CUDABlas.cpp` |
| cuDNN conv | `aten/src/ATen/native/cudnn/Conv_v7.cpp` |
| Attention | `aten/src/ATen/native/transformers/attention.cpp` |

Work through `modules/tensor` → `autograd` → `nn` → `cuda`, then open these files with the call stacks in the root README.
