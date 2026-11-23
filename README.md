# pytorch-basic-learning

Learn PyTorch by walking the **source tree**, the **dispatcher**, and small **repro cases** next to each module.

This is a draft map of how `torch/**` (Python), `torch/csrc` (bindings), `ATen`, and `c10` fit together — including where `torch.compile` sits — plus call stacks and starter cases under `modules/`.

Upstream paths below refer to a normal PyTorch checkout (e.g. `pytorch/torch`, `pytorch/aten`, `pytorch/c10`).

---

## 1. Big picture

```text
                        ┌─────────────────────────────────────────┐
                        │           Application / training         │
                        └─────────────────────┬───────────────────┘
                                              │
              ┌───────────────────────────────┴───────────────────────────────┐
              │                         Python layer                            │
              │  torch/**   nn  autograd  cuda  distributed  jit  export  ... │
              │                                                               │
              │  torch.compile path (graph):                                  │
              │    torch._dynamo  →  torch.fx  →  torch._inductor             │
              │         │ FX graph / guards          │ kernels / codegen      │
              └─────────┼────────────────────────────┼────────────────────────┘
                        │ eager ops                  │ may call back into ATen
                        ▼                            ▼
              ┌───────────────────────────────────────────────────────────────┐
              │              Python ↔ C++ interface                             │
              │                 torch/_C  (pybind / THP)                       │
              └─────────────────────────────┬─────────────────────────────────┘
                                            ▼
              ┌───────────────────────────────────────────────────────────────┐
              │                       C++ layer                                │
              │                    torch/csrc/**                               │
              │         autograd · cuda · distributed · jit · dynamo …         │
              │                            │                                   │
              │              ┌─────────────┴─────────────┐                     │
              │              ▼                           ▼                     │
              │           ATen                        c10                      │
              │   Tensor ops + Dispatcher      Tensor/Storage/Device           │
              │   native/CUDA/cuDNN/...        Allocator / Stream / Guard      │
              │              └─────────────┬─────────────┘                     │
              └────────────────────────────┼──────────────────────────────────┘
                                           ▼
              ┌───────────────────────────────────────────────────────────────┐
              │              Syscalls & drivers (CUDA / NCCL / …)               │
              └───────────────────────────────────────────────────────────────┘
```

**Eager vs compile (where `torch.compile` lives):**

| Path | Entry | What it does | Lands in |
| --- | --- | --- | --- |
| Eager | `torch.add` / `Tensor.method` | Python → `_C` → ATen `Dispatcher` → kernel | `aten` + `c10` |
| Compile | `torch.compile(fn)` | Capture frames in `_dynamo`, lower via `fx`, generate/run in `_inductor` | Still may call ATen ops / custom kernels; not a separate hardware layer |

```text
torch.compile(model)
        │
        ▼
torch/_dynamo          # bytecode → FX graph, guards, cache
        │
        ▼
torch/fx               # graph IR, passes
        │
        ▼
torch/_inductor        # scheduling, codegen (Triton/C++/CUDA), runtime
        │
        ├──► generated kernels
        └──► ATen / c10 for fallbacks & library ops
```

---

## 2. Top-level source map

| Tree | Role |
| --- | --- |
| `torch/` | Public Python API + compile stack (`_dynamo` / `_inductor` / `fx`) + thin wrappers |
| `torch/_C` | Extension module built from `torch/csrc` (the Python↔C++ bridge) |
| `torch/csrc/` | C++ frontend: bindings, autograd engine hooks, distributed, JIT, CUDA helpers |
| `aten/` | **A** **Ten**sor library: op schemas, `Dispatcher`, `native` CPU/CUDA/… kernels |
| `c10/` | Core runtime under ATen: `TensorImpl`, `Storage`, `Device`, `DispatchKey`, CUDA caching allocator / streams |
| `torchgen/` | Codegen for ATen ops / bindings from YAML |
| `test/` | Upstream unit tests (reference, not vendored here) |

### 2.1 `torch/` (Python) — directories worth knowing

| Directory | Module focus |
| --- | --- |
| `torch/_tensor.py`, `torch/storage.py` | Tensor / Storage Python surface |
| `torch/nn/`, `torch/nn/functional.py` | Modules + functional API |
| `torch/autograd/` | Autograd Python API, `Function`, anomaly mode |
| `torch/optim/` | Optimizers, LR schedulers |
| `torch/cuda/` | Device, streams, memory, AMP helpers (Python side) |
| `torch/distributed/` | Process groups, DDP/FSDP fronts, collectives |
| `torch/utils/data/` | `Dataset` / `DataLoader` |
| `torch/jit/`, `torch/export/` | Script/trace and newer export |
| `torch/fx/` | Graph IR used by compile & export |
| `torch/_dynamo/` | `torch.compile` frontend (frame evaluation, guards) |
| `torch/_inductor/` | `torch.compile` backend (codegen / runtime) |
| `torch/_dispatch/`, `torch/library.py` | Python view of dispatcher / custom ops |
| `torch/backends/` | cuDNN / MKL / CUDA backend toggles |
| `torch/amp/` | Autocast / GradScaler |
| `torch/profiler/` | Profiling APIs |
| `torch/quantization/`, `torch/ao/` | Quantization stacks |
| `torch/sparse/`, `torch/nested/`, `torch/linalg/` | Specialized tensor domains |

### 2.2 `torch/csrc/` (C++ frontend)

| Directory | Module focus |
| --- | --- |
| `torch/csrc/` (root) | `Tensor`, `Storage`, `Stream`, `Module.cpp` pybind entry |
| `torch/csrc/autograd/` | Engine, `Variable`, saved tensors, Python `Function` bridge |
| `torch/csrc/cuda/` | CUDA module bindings, generators, IPC |
| `torch/csrc/distributed/c10d/` | `ProcessGroup*`, NCCL/Gloo, work/futures |
| `torch/csrc/jit/` | TorchScript interpreter / freeze / mobile bits |
| `torch/csrc/dynamo/`, `inductor/` | C++ helpers for compile stack |
| `torch/csrc/tensor/` | Python tensor methods bound into `_C` |
| `torch/csrc/profiler/` | Kineto / activity APIs |

### 2.3 `aten/` and `c10/`

| Path | Module focus |
| --- | --- |
| `aten/src/ATen/core/dispatch/` | **Dispatcher**, `OperatorEntry`, boxed/unboxed call |
| `aten/src/ATen/native/` | Default CPU (and many structured) kernel implementations |
| `aten/src/ATen/native/cuda/` | CUDA kernels (e.g. elementwise, reductions) |
| `aten/src/ATen/cuda/` | CUDABlas, CUDASparse, device helpers (`CUDABlas.cpp` → gemm) |
| `aten/src/ATen/native/cudnn/` | cuDNN wrappers (`Conv_v7.cpp` → conv) |
| `aten/src/ATen/native/transformers/` | Attention / SDPA related (`attention.cpp`, flash paths) |
| `c10/core/` | `TensorImpl`, `StorageImpl`, `Device`, `DispatchKey(Set)`, guards |
| `c10/cuda/` | `CUDACachingAllocator`, `CUDAStream`, `CUDAGuard` |

---

## 3. Dispatcher (the spine of eager ops)

Almost every eager op eventually hits:

```text
Python API
  → torch/_C binding  (torch/csrc/…)
  → at::Tensor / at::native wrapper
  → c10::Dispatcher::call / callBoxed
  → pick KernelFunction by DispatchKeySet
  → ATen native / backend kernel
  → c10 CUDA allocator / stream / driver
```

### 3.1 DispatchKey (why backends fork)

Defined mainly in `c10/core/DispatchKey.h`. A tensor carries a key set (e.g. `CPU`, `CUDA`, `Autograd`, `Functionalize`, `Python`, …). The dispatcher masks and prioritizes keys to choose **one** kernel.

Mental model:

```text
Tensor on CUDA + requires_grad
  DispatchKeySet ≈ { AutogradCUDA, CUDA, … }
        │
        ▼
  Autograd kernel wraps the math kernel
        │
        ▼
  CUDA math kernel (ATen native/cuda or library)
```

Custom backends / privateuse1 plug in extra keys the same way.

### 3.2 Core call stack — `torch.add` (eager, CUDA)

Simplified stack (names vary slightly by version):

```text
torch.add(a, b) / a + b                          # torch/_tensor*.py / _ops
        │
        ▼
THPVariable_add / generated binding              # torch/csrc/autograd/generated or tensor API
        │
        ▼
at::add(a, b)                                    # ATen API
        │
        ▼
c10::Dispatcher::call<…>(op, a, b)               # aten/.../dispatch/Dispatcher.h
        │  resolve OperatorHandle "aten::add.Tensor"
        │  compute DispatchKeySet from args
        ▼
KernelFunction::call                             # e.g. Autograd → then CUDA
        │
        ▼
at::native::add_cuda / structured cuda kernel    # aten/src/ATen/native/cuda/…
        │
        ▼
c10::cuda::CUDACachingAllocator + current stream # c10/cuda/…
        │
        ▼
CUDA driver / runtime
```

### 3.3 Autograd wrap

```text
Dispatcher sees Autograd* key first
        │
        ▼
autograd node (AddBackward0, …)                  # torch/csrc/autograd/…
        │  forward: call redispatched math kernel
        │  save tensors for backward
        ▼
Engine::execute on loss.backward()               # torch/csrc/autograd/engine.cpp
```

### 3.4 `nn.Module` forward (high level)

```text
model.forward / __call__                         # torch/nn/modules/module.py
        │  hooks, training mode
        ▼
F.linear / F.conv2d / …                          # torch/nn/functional.py
        │
        ▼
same Dispatcher path as §3.2
```

### 3.5 CUDA memory / stream touchpoints

```text
op kernel launch
        │
        ▼
getCurrentCUDAStream(device)                     # c10/cuda/CUDAStream.cpp
cudaMalloc / caching allocate                    # c10/cuda/CUDACachingAllocator.cpp
CUDAGuard / OptionalCUDAGuard                    # c10/cuda/CUDAGuard.h
```

### 3.6 Distributed collective (NCCL)

```text
torch.distributed.all_reduce(t)                  # torch/distributed/distributed_c10d.py
        │
        ▼
ProcessGroupNCCL::allreduce                      # torch/csrc/distributed/c10d/ProcessGroupNCCL.*
        │
        ▼
ncclAllReduce(...)                               # NCCL / vendor CCL
```

See also the eight smoke tests under `modules/distributed/` (and the longer write-up style in local notes on ProcessGroupNCCL).

### 3.7 `torch.compile` vs dispatcher

Compile does **not** replace the dispatcher as a concept; it often:

1. Captures Python frames → FX graph (`_dynamo`).
2. Lowers / fuses / generates kernels (`_inductor`).
3. Still calls into ATen (dispatcher) for unimplemented / library ops, or emits kernels that bypass per-op Python overhead.

```text
compiled_fn(*args)
        │
        ▼
inductor runner / cudagraphs (optional)
        │
        ├──► fused Triton/CUDA kernel
        └──► at::ops… → Dispatcher (fallback / decomps)
```

---

## 4. Repo layout (learning modules)

Each directory has a short README and starter cases. Cases are **educational smoke tests**, not a full PyTorch `test/` mirror.

```text
pytorch-basic-learning/
├── README.md                 # this file
├── docs/dispatcher/          # deeper call-stack notes
└── modules/
    ├── tensor/               # storage, stride, view vs copy
    ├── autograd/             # Function, grad_fn, backward
    ├── nn/                   # Module / functional
    ├── cuda/                 # device, stream, allocator signals
    ├── distributed/          # NCCL collective smoke tests (8-rank)
    ├── compile/              # torch.compile / dynamo basics
    ├── export/               # export / fx peek
    └── c10_aten/             # reading notes tying cases → source paths
```

| Module dir | Learn | Starter entry |
| --- | --- | --- |
| `modules/tensor` | Contiguity, view/reshape/clone | `python case_view_copy.py` |
| `modules/autograd` | Custom `Function`, grad check | `python case_custom_function.py` |
| `modules/nn` | Parameter registration, forward | `python case_linear_forward.py` |
| `modules/cuda` | Stream, mem stats | `python case_stream_mem.py` |
| `modules/distributed` | Eight collectives | `./run_allreduce.sh` (needs 8 GPUs) |
| `modules/compile` | compile vs eager | `python case_compile_basic.py` |
| `modules/export` | `torch.export` sketch | `python case_export_basic.py` |

---

## 5. How to use

1. Read the diagram in §1 and pick a module in §2.
2. Open the matching `modules/<name>/README.md`.
3. Run the starter case; then jump to the listed upstream files with the §3 stacks.
4. Extend cases as you dig (dispatcher keys, compile counters, NCCL traces, …).

```bash
git clone git@github.com:HangJie720/pytorch-basic-learning.git
cd pytorch-basic-learning/modules/tensor
python case_view_copy.py
```

---

## 6. Status

**Draft v0.** Next iterations can deepen per-op stacks (conv, SDPA, FSDP), add DispatchKey dumps in cases, and wire more of `aten/native` reading guides under `modules/c10_aten/`.
