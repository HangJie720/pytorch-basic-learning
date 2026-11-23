# distributed

Eight 8-rank smoke tests for NCCL collective primitives (AllReduce, Broadcast, Reduce, AllGather, ReduceScatter, AlltoAll, Gather, Scatter).

These cases also live under `/home/jie.hang/tests/python/distributed` for ad-hoc runs outside this repo.

## Upstream map

| Topic | Path |
| --- | --- |
| Python API | `torch/distributed/distributed_c10d.py` |
| ProcessGroupNCCL | `torch/csrc/distributed/c10d/ProcessGroupNCCL.*` |

## Call stack

```text
dist.all_reduce / broadcast / …
  → c10d ProcessGroup
  → ProcessGroupNCCL
  → ncclAllReduce / ncclBroadcast / …
```

## How to run

```bash
cd modules/distributed
./run_allreduce.sh
./run_broadcast.sh --N 4096 --root 2 --dtype float16
```

Needs 8 GPUs. `NPROC_PER_NODE` defaults to 8. `root` is a **rank id**, not a device id.

| Case | What it checks |
| --- | --- |
| `test_allreduce.py` | Full reduced result on every rank (DDP-style) |
| `test_broadcast.py` | Root buffer copied to all ranks |
| `test_reduce.py` | Reduced result only on root |
| `test_allgather.py` | Concatenation `k*N` on every rank |
| `test_reducescatter.py` | Reduced then split; rank `Y` keeps chunk `Y` |
| `test_alltoall.py` | Rank `Y` segment `X` → rank `X` |
| `test_gather.py` | Concatenation only on root |
| `test_scatter.py` | Root splits `k*N` across ranks |

Identities: `Reduce + Broadcast = AllReduce`, `ReduceScatter + AllGather = AllReduce`.

## Reference

[03_看懂多卡分布式训练：一文读懂NCCL八大集合通信原语](https://mp.weixin.qq.com/s?__biz=MzkxNDQ5MTI2OA==&mid=2247487813&idx=1&sn=de38842ba5735b88ab837d2512bea005&chksm=c16cc3aef61b4ab8a976f43fdd8d39a861431acc95eca7f3bfe30ab53b8e02b55c7dfc68cd8a&cur_album_id=4770922934775611392&scene=189#wechat_redirect)
