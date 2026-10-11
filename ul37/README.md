# UL37 benchmark runner

`run_ul37.py` fine-tunes GTR on each of the 37 UL37 detection datasets and saves the predictions the benchmark scores.
Per dataset it converts the benchmark's pinned YOLO dataset to COCO json, starts from the released COCO detector
without its class head, and trains 100 epochs through `train.sh` at total batch 16 (seed 0, FP32 with TF32, 640 square)
from `configs/det/coco_finetune/gtr_<size>.yml` with the LR scaled linearly to batch 16 and every epoch field rescaled
from 30 to 100 epochs. Micro-batch shrinks with the densest image's box count and gradient accumulation keeps 16 images
per step. It then writes `predictions.json` (top 300 from the final EMA weights), `last_ema.pth` and `args.yaml`. A
dataset with `args.yaml` is skipped only when its saved provenance, this repo's commit included, matches the current run.

## Setup

`setup_gtr_env.sh` runs `uv sync --frozen`, builds the fused GLA kernel for sm_120 with the nvcc in `/usr` (its CUDA
version must equal torch's, 12.8) and runs `test_gla.py` on the given GPU, which must print `ALL PASS`. `check_ul37.py`
then checks that each size's init covers every tensor but the class head, and with `--datasets` converts and counts
every UL37 dataset, which the first run of a protocol needs.

```bash
git clone --single-branch -b ul37 git@github.com:fcakyon/GTR.git gtr && cd gtr
git checkout SHA && bash ul37/setup_gtr_env.sh [GPU]
CUDA_VISIBLE_DEVICES= .venv/bin/python ul37/check_ul37.py --benchmark BENCH --weights WEIGHTS [--datasets]
```

## Run

`BENCH` is a UL37 benchmark checkout: runs read its `ul37_platform_uris.txt`, `ul37_datasets.txt` and `.venv-ul37`
runtime with `.config-ul37-081026`, and scoring runs its `val_ultrabench.py`. `WEIGHTS` holds the Hugging
Face `Phoenix8125/GTR` detectors from its `det/` dir, `gtr_{s,m,l,x}_coco.pth`.

```bash
.venv/bin/python ul37/run_ul37.py SIZE --project RUNS --benchmark BENCH --weights WEIGHTS --device 0
```

Results go to `RUNS/ultrabench-081026-gtr-SIZE/multitrain`. An unsharded run scores itself into
`val_ultrabench.json`. For shards, give each process `--subset START:STOP` (or comma-separated dataset names) and the
same `--name`, then score once with `--val-only` after all shards finish.

## Latency export

`export_gtr.sh GPU STAGE` exports FP16 ONNX with GLA as plain ONNX ops (`gtr_stable.py`) and FP32 graph I/O
(`io_fp32.py`). `gtr_ref.py` saves the PyTorch FP32 reference outputs for the TensorRT parity check. Both run with
this checkout's `.venv` plus onnx, onnxsim and onnxruntime.

## COCO-SAMA labels

COCO-SAMA re-annotates COCO val2017 images. Build its `instances_val2017.json` from Sama's release:

```bash
curl -LO https://sama-documentation-assets.s3.amazonaws.com/sama-coco/sama-coco-val.zip
echo "c55454f6657ce63891783136cc5dfe5f1e9c5e1748c0d8ca08b456a804eab15f  sama-coco-val.zip" | sha256sum -c
unzip -j sama-coco-val.zip '*sama_coco_coco_format_val_*.json'
python3 -c "import json; s = [json.load(open(f'sama_coco_coco_format_val_{i}.json')) for i in range(5)]; open('instances_val2017.json', 'w').write(json.dumps({k: s[0][k] for k in ('info', 'licenses', 'categories')} | {'images': sorted((x for d in s for x in d['images']), key=lambda x: x['id']), 'annotations': [x for d in s for x in d['annotations']]}))"
sha256sum instances_val2017.json # c8575a4b8eb39a8f7316c3e6f51349834ca0dc3b11e35f615497064c5d0b79b0
```
