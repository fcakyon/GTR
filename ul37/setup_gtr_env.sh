#!/bin/bash
# Prepare this GTR clone for UL37: .venv from its uv.lock and the sm_120 fused GLA kernel. Check out the pinned commit
# first. Usage: ul37/setup_gtr_env.sh [GPU index for test_gla.py] (idempotent)
set -euo pipefail
cd "$(dirname "$0")/.."
DIR=$PWD
PATH=$HOME/.local/bin:$PATH uv sync -q --frozen
# The extension build refuses an nvcc whose CUDA version differs from torch's. It uses the nvcc in /usr.
NVCC=$(nvcc --version | sed -n 's/.*release \([0-9.]*\),.*/\1/p')
TORCH_CUDA=$(.venv/bin/python -c 'import torch; print(torch.version.cuda)')
[ "$NVCC" = "$TORCH_CUDA" ] || {
  echo "nvcc $NVCC != torch CUDA $TORCH_CUDA"
  exit 1
}
CSRC=engine/gtr/backbone/csrc
ls $CSRC/gla_torch_ext*.so > /dev/null 2>&1 || CUDA_HOME=/usr TORCH_CUDA_ARCH_LIST=12.0 PATH=$DIR/.venv/bin:$PATH \
  PY=$DIR/.venv/bin/python bash $CSRC/build.sh > ul37/build_gla.log 2>&1
sha256sum $CSRC/gla_torch_ext*.so
[ $# -eq 0 ] || (cd $CSRC && CUDA_VISIBLE_DEVICES=$1 $DIR/.venv/bin/python test_gla.py > $DIR/ul37/test_gla.log)
