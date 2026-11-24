# distributed

PyTorch distributed learning cases, split by topic so collectives, symmetric memory, and later modules stay separate.

| Subdir | Focus | Entry |
| --- | --- | --- |
| [`collectives/`](collectives/) | Eight NCCL collective smoke tests (AllReduce, Broadcast, Reduce, AllGather, ReduceScatter, AlltoAll, Gather, Scatter) | `cd collectives && ./run_allreduce.sh` |
| `symmetric/` *(planned)* | Symmetric memory / NVSHMEM-style paths | — |

Call stack overview:

<img src="../../docs/figures/distributed_call_stack.jpg" width="640" alt="Distributed collective call stack" />
