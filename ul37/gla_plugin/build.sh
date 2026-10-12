#!/bin/bash
# Compile GTR's GatedLinearAttention TensorRT plugin for TensorRT 11.2 on sm_120 with the patched gla_chunk.cuh next to
# this file, through GTR's own tensorrt_plugin/trt_plugin/build.sh (gla_plugin.cu unchanged). Writes
# tensorrt_plugin/trt_plugin/libgla_plugin.so. Usage: ul37/gla_plugin/build.sh TRT_PY TRT_INCLUDE
#   TRT_PY       python of the environment that runs the engines (tensorrt 11.2 with a cu13 torch). Its
#                tensorrt_libs/libnvinfer.so.11 and nvidia/cu13/lib/libcudart.so.13 are linked and become the rpath.
#   TRT_INCLUDE  TensorRT 11.2 headers (NvInfer.h), which the tensorrt pip wheels do not ship.
# nvcc is the pip CUDA 13.0 compiler in a uv venv under $GLA_BUILD (default ~/.cache/gtr-gla-plugin), made on first use.
set -euo pipefail
PY=$1 INC=$2
HERE=$(cd "$(dirname "$0")" && pwd)
B=${GLA_BUILD:-$HOME/.cache/gtr-gla-plugin}
SITE=$("$PY" -c "import tensorrt_libs, pathlib; print(pathlib.Path(tensorrt_libs.__file__).parent.parent)")
[ -x "$B/nvcc/bin/python" ] || { PATH=$HOME/.local/bin:$PATH uv venv -q "$B/nvcc" --python 3.13 \
  && PATH=$HOME/.local/bin:$PATH uv pip install -q --python "$B/nvcc/bin/python" nvidia-cuda-nvcc==13.0.88 \
    nvidia-cuda-crt==13.0.88 nvidia-nvvm==13.0.88 nvidia-cuda-runtime==13.0.96; }
mkdir -p "$B/trt/lib"
ln -sfn "$INC" "$B/trt/include"
for f in tensorrt_libs/libnvinfer.so.11 nvidia/cu13/lib/libcudart.so.13; do
  ln -sf "$SITE/$f" "$B/trt/lib/$(basename "$f")"
  ln -sf "$SITE/$f" "$B/trt/lib/$(basename "$f" | cut -d. -f1-2)"
done
TRT_ROOT=$B/trt CUDA_HOME=$B/nvcc/lib/python3.13/site-packages/nvidia/cu13 GLA_SRC=$HERE SM=120 \
  bash "$HERE/../../tensorrt_plugin/trt_plugin/build.sh"
