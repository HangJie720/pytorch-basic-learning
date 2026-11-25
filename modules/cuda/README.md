# cuda

## Diagrams

<img src="images/cuda_call_stack.jpg" width="640" alt="CUDA runtime call stack" />

<img src="images/stream_stack.png" width="640" alt="stream stack" />

<img src="images/p2p_vs_copy.png" width="640" alt="P2P vs copy" />

## Cases

| Case | Idea |
| --- | --- |
| `case_stream_mem.py` | explicit stream + memory stats |
| `case_peer_access.py` | print CUDA P2P matrix (for symmetric mem) |

```bash
cd modules/cuda
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python case_stream_mem.py
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python case_peer_access.py
```

## Upstream

| Topic | Path |
| --- | --- |
| Streams | `c10/cuda/CUDAStream.cpp` |
| Guards | `c10/cuda/CUDAGuard.h` |

Allocator details: [`../allocator/`](../allocator/).  
Symmetric memory / P2P: [`../distributed/symmetric/`](../distributed/symmetric/).
