# distributed

PyTorch distributed learning cases, **one subdir per topic**.

| Subdir | Focus | Entry |
| --- | --- | --- |
| [`collectives/`](collectives/) | Eight NCCL collective smoke tests | `cd collectives && ./run_allreduce.sh` |
| [`symmetric/`](symmetric/) | Symmetric memory put/get + signal | `cd symmetric && ./run_put_get.sh` (2 GPUs, P2P) |
| [`fsdp/`](fsdp/) | Fully Sharded Data Parallel (notes + diagram) | README |
| [`device_mesh/`](device_mesh/) | DeviceMesh / DTensor | README |
| [`pipelining/`](pipelining/) | Pipeline parallelism | README |
| [`rpc/`](rpc/) | RPC + distributed autograd | README |
| [`checkpoint/`](checkpoint/) | Distributed checkpoint | README |

Call stack (collectives path):

<img src="../../docs/figures/distributed_call_stack.jpg" width="640" alt="Distributed collective call stack" />

Start with **collectives**, then **symmetric** to contrast two-sided vs one-sided communication. FSDP/mesh/pipeline build on those primitives.
