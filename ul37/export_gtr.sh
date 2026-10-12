#!/bin/bash
# Usage: export_gtr.sh GPU STAGE [sizes...]
# Export trained GTR COCO detectors (STAGE/weights/gtr/gtr_<size>_coco.pth) to static batch-1 FP16 ONNX with FP32 graph
# I/O at 640 into STAGE/gtr-onnx, through gtr_stable.py and io_fp32.py next to this file, with the .venv of the GTR
# checkout holding this file plus onnx, onnxsim and onnxruntime. With PLUGIN=1 GTR's own tensorrt_plugin/export_onnx.py
# exports GLA as GatedLinearAttention plugin nodes instead (gtr-<size>-plugin.onnx, for gla_plugin/build.sh's plugin).
# Fails closed unless the physical GPU has no compute process.
set -u
G=$1 N=$2
shift 2
HERE=$(cd "$(dirname "$0")" && pwd)
GTR=$(dirname "$HERE")
O=$N/gtr-onnx
mkdir -p "$O"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$G
UUID=$(nvidia-smi -i "$G" --query-gpu=uuid --format=csv,noheader)
echo "[$(date -u +%FT%TZ)] gpu=$G $UUID gtr=$(git -C "$GTR" rev-parse HEAD)"
if nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep -q "$UUID"; then
  echo "ABORT: GPU $G busy"
  exit 1
fi
cd "$GTR/tensorrt_plugin" || exit 1
for s in ${@:-s m l x}; do
  ck=$N/weights/gtr/gtr_${s}_coco.pth
  echo "EXPORT gtr-$s ckpt md5 $(md5sum "$ck" | cut -d' ' -f1)"
  n=gtr-$s${PLUGIN:+-plugin}
  if [ -n "${PLUGIN:-}" ]; then
    cmd=(export_onnx.py --dtype fp16)
  else
    cmd=("$HERE/gtr_stable.py" --dtype fp16 --no-fuse-gate)
  fi
  "$GTR/.venv/bin/python" "${cmd[@]}" -c "../configs/det/coco_finetune/gtr_$s.yml" -o "$O/$n.fp16io.onnx" \
    --resume "$ck" > "$O/$n.export.log" 2>&1 || {
    echo "FAIL $n"
    tail -20 "$O/$n.export.log"
    exit 1
  }
  "$GTR/.venv/bin/python" "$HERE/io_fp32.py" "$O/$n.fp16io.onnx" "$O/$n.onnx" || exit 1
  grep -E "loaded|patched|output shapes|GatedLinear|final graph" "$O/$n.export.log"
done
echo DONE
