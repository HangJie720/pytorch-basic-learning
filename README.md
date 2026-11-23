# pytorch-basic-learning

Learn PyTorch by walking the **source tree**, the **dispatcher**, and small **repro cases** next to each module.

This is a draft map of how `torch/**` (Python), `torch/csrc` (bindings), `ATen`, and `c10` fit together — including where `torch.compile` sits — plus call stacks and starter cases under `modules/`.

Upstream paths below refer to a normal PyTorch checkout (e.g. `pytorch/torch`, `pytorch/aten`, `pytorch/c10`).

Flowcharts live under [`docs/figures/`](docs/figures/).

---

## 1. Big picture

![PyTorch composition structure](docs/figures/pytorch_architecture.jpg)

| Layer | Path | Role |
| --- | --- | --- |
| Python | `torch/**` | Public API: `nn`, `autograd`, `cuda`, `distributed`, `jit`, `export`, … |
| Compile (in Python) | `_dynamo` → `fx` → `_inductor` | Graph capture / lowering / codegen for `torch.compile` |
| Native interface | `torch/_C` | pybind / THP bridge into C++ |
| C++ frontend | `torch/csrc` | Bindings, autograd engine hooks, c10d, JIT, CUDA helpers |
| Libraries | `ATen` + `c10` | Ops + **Dispatcher**; Tensor/Storage/Device/allocator/stream |
| Runtime | Syscalls & drivers | CUDA, NCCL, … |

**Eager vs compile**

| Path | Entry | What it does | Lands in |
| --- | --- | --- | --- |
| Eager | `torch.add` / `Tensor.method` | Python → `_C` → ATen `Dispatcher` → kernel | `aten` + `c10` |
| Compile | `torch.compile(fn)` | Capture in `_dynamo`, lower via `fx`, run in `_inductor` | Generated kernels and/or ATen fallbacks |

![torch.compile path](docs/figures/torch_compile_flow.jpg)

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

Almost every eager op eventually hits the stack below. See also [`docs/dispatcher/overview.md`](docs/dispatcher/overview.md).

![Eager op dispatcher stack](docs/figures/dispatcher_eager_stack.jpg)

### 3.1 DispatchKey (why backends fork)

Defined mainly in `c10/core/DispatchKey.h`. A tensor carries a key set (e.g. `CPU`, `CUDA`, `Autograd`, `Functionalize`, `Python`, …). The dispatcher masks and prioritizes keys to choose **one** kernel.

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

### 3.2 Notes on the eager stack

| Step | Typical landing |
| --- | --- |
| Python API | `torch/_tensor*.py`, `torch/_ops` |
| Binding | `torch/csrc` / generated THP methods |
| ATen API | `at::add` etc. |
| Dispatcher | `aten/src/ATen/core/dispatch/Dispatcher.h` |
| Kernel | Autograd wrap → `at::native::*` / CUDA / library |
| Runtime | `c10/cuda/CUDACachingAllocator`, `CUDAStream`, driver |

### 3.3 Autograd wrap

```text
Dispatcher sees Autograd* key first
        │
        ▼
autograd node (AddBackward0, …)     # torch/csrc/autograd/…
        │  forward: redispatched math kernel
        ▼
Engine::execute on loss.backward()  # torch/csrc/autograd/engine.cpp
```

### 3.4 `nn.Module` forward

```text
module(x) → forward hooks → forward() → F.linear / F.conv2d → same Dispatcher path as §3.2
```

### 3.5 CUDA memory / stream touchpoints

```text
op kernel launch
  → getCurrentCUDAStream            # c10/cuda/CUDAStream.cpp
  → CUDACachingAllocator            # c10/cuda/CUDACachingAllocator.cpp
  → CUDAGuard                       # c10/cuda/CUDAGuard.h
```

### 3.6 Distributed collective (NCCL)

![Distributed collective call stack](docs/figures/distributed_call_stack.jpg)

See the eight smoke tests under [`modules/distributed/`](modules/distributed/) (each collective has an official NCCL diagram + case description).

### 3.7 `torch.compile` vs dispatcher

Compile does **not** replace the dispatcher. It captures Python → FX (`_dynamo`), lowers/fuses (`_inductor`), and still calls into ATen for fallbacks / library ops (see the compile flowchart in §1).

---

## 4. Repo layout (learning modules)

```text
pytorch-basic-learning/
├── README.md
├── docs/
│   ├── figures/              # architecture & call-stack flowcharts
│   └── dispatcher/           # deeper dispatcher notes
└── modules/
    ├── tensor/ · autograd/ · nn/ · cuda/
    ├── distributed/ · compile/ · export/
    └── c10_aten/
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

1. Skim the figures in §1 / §3 (or browse [`docs/figures/`](docs/figures/)).
2. Pick a module in §2 and open `modules/<name>/README.md`.
3. Run the starter case; jump to the listed upstream files with the stacks above.
4. Extend cases as you dig (DispatchKey dumps, compile counters, NCCL traces, …).

```bash
git clone git@github.com:HangJie720/pytorch-basic-learning.git
cd pytorch-basic-learning/modules/tensor
python case_view_copy.py
```

---

## 6. Status

**Draft v0.** Flowcharts are under `docs/figures/` (JPG + editable SVG). Next: deeper per-op stacks (conv, SDPA, FSDP) and richer DispatchKey dumps in cases.
