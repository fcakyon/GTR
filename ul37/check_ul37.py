"""Check the GTR UL37 environment, init coverage and optionally every UL37 dataset."""

import argparse
import json
import sys
import tempfile
from pathlib import Path

import run_ul37 as runner
import torch
import yaml
from engine.core import YAMLConfig
from engine.gtr.backbone import vit_adapter

parser = argparse.ArgumentParser()
parser.add_argument("--benchmark", type=Path, required=True, help="UL37 benchmark checkout with val_ultrabench.py")
parser.add_argument("--weights", type=Path, required=True, help="Dir of Phoenix8125/GTR det/gtr_<size>_coco.pth")
parser.add_argument("--datasets", action="store_true", help="Also convert and count every UL37 dataset")
args = parser.parse_args()

assert sys.version_info[:2] == (3, 11)
assert vit_adapter._gla_cuda_ext is not None, "fused GLA kernel missing"
with tempfile.TemporaryDirectory() as scratch:
    for size in runner.SIZES:  # the init must cover every tensor except the class-shaped head of a 3-class model
        config = Path(scratch) / f"{size}.yml"
        config.write_text(
            yaml.safe_dump(
                {"__include__": [str(runner.GTR / f"configs/det/coco_finetune/gtr_{size}.yml")], "num_classes": 3}
            )
        )
        model = YAMLConfig(str(config), ViTAdapterSpatialSwiGLU={"skip_weights_warning": True}).model.state_dict()
        init = torch.load(runner.build_init(size, args.weights), map_location="cpu", weights_only=True)["model"]
        assert all(init[k].shape == v.shape for k, v in model.items() if k in init), size
        missing = model.keys() - init.keys()
        assert missing and all(k.startswith(runner.HEAD) for k in missing), size
        assert not init.keys() - model.keys(), size
        print(size, json.dumps({k: v for k, v in runner.recipe(size, 100).items() if k in ("optimizer", "lr_scale")}))

fraction, roster, data_yamls, ultralytics_sha = runner.load_roster(args.benchmark)
print(json.dumps({"fraction": fraction, "datasets": len(roster), "ultralytics_sha": ultralytics_sha}))

if args.datasets:
    counts = {}
    for name, data_yaml in data_yamls.items():
        with tempfile.TemporaryDirectory() as scratch:
            classes = runner.benchmark_json(args.benchmark, runner.TO_COCO, str(data_yaml), scratch)
            splits = {split: json.loads(Path(scratch, f"{split}.json").read_text()) for split in ("train", "val")}
        counts[name] = {"classes": classes, **{split: len(coco["images"]) for split, coco in splits.items()}}
        assert counts[name]["train"] >= runner.BATCH, (name, counts[name])  # GTR drops the last partial batch
    print(json.dumps(counts, sort_keys=True))
