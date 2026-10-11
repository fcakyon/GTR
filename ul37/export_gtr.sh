#!/bin/bash
# Usage: export_gtr.sh GPU STAGE [sizes...]
# Export trained GTR COCO detectors (STAGE/weights/gtr/gtr_<size>_coco.pth) to static batch-1 FP16 ONNX with FP32 graph
# I/O at 640 into STAGE/gtr-onnx, through gtr_stable.py and io_fp32.py next to this file, with the .venv of the GTR
# checkout holding this file plus onnx, onnxsim and onnxruntime. Fails closed unless the physical GPU has no compute
# process.
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
  "$GTR/.venv/bin/python" "$HERE/gtr_stable.py" -c "../configs/det/coco_finetune/gtr_$s.yml" -o "$O/gtr-$s.fp16io.onnx" \
    --dtype fp16 --resume "$ck" --no-fuse-gate > "$O/gtr-$s.export.log" 2>&1 || {
    echo "FAIL gtr-$s"
    tail -20 "$O/gtr-$s.export.log"
    exit 1
  }
  "$GTR/.venv/bin/python" "$HERE/io_fp32.py" "$O/gtr-$s.fp16io.onnx" "$O/gtr-$s.onnx" || exit 1
  grep -E "loaded|patched|output shapes|GatedLinear|final graph" "$O/gtr-$s.export.log"
done
echo DONE
