#!/usr/bin/env bash
# 运行 8-rank ReduceScatter(sum) case
# 用法:
#   ./run_reducescatter.sh
#   ./run_reducescatter.sh --N 4096 --dtype float16

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29504}"

torchrun \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  test_reducescatter.py \
  "$@"
