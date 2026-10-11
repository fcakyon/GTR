"""Save GTR's PyTorch FP32 reference outputs (unpatched deploy model, fla chunk_gla) for the TensorRT parity check.

Inputs are verify_trt.py's seed-0 noise plus the given images, keyed by file stem: Ultralytics' bus.jpg and zidane.jpg
and COCO val2017 000000000139.jpg and 000000000285.jpg, resized to 640x640 and ImageNet-normalized. Writes
<out>/gtr-<size>.ref.npz with every input and its pred_logits / pred_boxes. The GTR repo is the checkout holding this
file.

Usage: CUDA_VISIBLE_DEVICES=<gpu> .venv/bin/python ul37/gtr_ref.py <out dir> <weights dir> <sizes, e.g. smlx> <image>...
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision

GTR = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(GTR), str(GTR / "tensorrt_plugin")]
from export_onnx import _IMAGENET_MEAN, _IMAGENET_STD, build_model

out, weights, sizes = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
images = [Path(p) for p in sys.argv[4:]]
mean, std = (torch.tensor(t, device="cuda").view(1, 3, 1, 1) for t in (_IMAGENET_MEAN, _IMAGENET_STD))
for s in sizes:
    torch.manual_seed(0)
    model = build_model(str(GTR / f"configs/det/coco_finetune/gtr_{s}.yml"), resume=str(weights / f"gtr_{s}_coco.pth"))[
        0
    ]
    model = model.cuda().eval().float()
    for m in model.modules():
        if hasattr(m, "_deploy_parallel_stages"):
            m._deploy_parallel_stages = False
    torch.manual_seed(0)
    xs = {"randn": torch.randn(1, 3, 640, 640, device="cuda")}
    for p in images:
        img = torchvision.io.read_image(str(p), torchvision.io.ImageReadMode.RGB).cuda()[None].float() / 255
        xs[p.stem] = (F.interpolate(img, (640, 640), mode="bilinear", align_corners=False) - mean) / std
    arrays = {}
    with torch.no_grad():
        for name, x in xs.items():
            y = model(x)
            arrays |= {
                f"{name}/images": x,
                f"{name}/pred_logits": y["pred_logits"],
                f"{name}/pred_boxes": y["pred_boxes"],
            }
            top = y["pred_logits"].sigmoid().flatten(1).topk(5)
            print(
                s,
                name,
                [(round(v, 3), c % 80) for v, c in zip(top.values[0].tolist(), top.indices[0].tolist())],
                flush=True,
            )
    np.savez(out / f"gtr-{s}.ref.npz", **{k: v.float().cpu().numpy() for k, v in arrays.items()})
print("torch", torch.__version__)
