# distributed

PyTorch distributed learning cases, **one subdir per topic**.

## Contents

- [collectives](collectives/) — eight NCCL collective smoke tests
- [symmetric](symmetric/) — symmetric memory put/get + signal (2 GPUs, P2P)
- [fsdp](fsdp/) — Fully Sharded Data Parallel
- [device_mesh](device_mesh/) — DeviceMesh / DTensor
- [pipelining](pipelining/) — pipeline parallelism
- [rpc](rpc/) — RPC + distributed autograd
- [checkpoint](checkpoint/) — distributed checkpoint

Call stack (collectives path):

<img src="../../docs/figures/distributed_call_stack.jpg" width="640" alt="Distributed collective call stack" />

Start with **collectives**, then **symmetric** to contrast two-sided vs one-sided communication. FSDP/mesh/pipeline build on those primitives.
