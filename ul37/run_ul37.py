#!/usr/bin/env python3
"""Train GTR on Platform-backed UL37 datasets and save canonical prediction artifacts."""

from __future__ import annotations

import argparse
import gc
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

GTR = Path(__file__).resolve().parents[1]  # runs from <repo>/ul37
sys.path.insert(0, str(GTR))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ul37_common import (
    HEAD,
    PROTOCOL,
    TO_COCO,
    benchmark_json,
    check_init_load,
    grad_accum_steps,
    load_roster,
    lock_dataset,
    pending_datasets,
    run_benchmark,
    save_detr_predictions,
    sha256,
    write_args,
)

os.environ.update(GTR_TF32="1", GTR_FAST_MAL_LOSS="1", GTR_SANITIZE_LOSSES="0", OPENCV_NUM_THREADS="1")  # train.sh

SIZES = "smlx"
RUNNER_SHA256 = sha256(Path(__file__))  # at import: the code this process runs, even if redeployed under it
BATCH, SEED, IMGSZ, MAX_DET = 16, 0, 640, 300


def recipe(size: str, epochs: int) -> dict:
    """Return the UL37 overrides of the repo's COCO finetune config, rescaled to the epochs and batch 16."""
    from engine.core.yaml_utils import load_config

    reference = load_config(str(GTR / f"configs/det/coco_finetune/gtr_{size}.yml"))
    train = reference["train_dataloader"]
    transforms, collate = train["dataset"]["transforms"], train["collate_fn"]
    assert reference["flat_epoch"] is None and reference["lrsheduler"] == "flatcosine", reference
    lr_scale = BATCH / train["total_batch_size"]  # linear LR scaling from the config's global batch

    def epoch(value: int) -> int:
        return round(value * epochs / reference["epochs"])

    return {
        "epochs": epochs,
        "checkpoint_freq": epochs,  # last.pth only, no 370 MB checkpoint per epoch
        "optimizer": {
            "lr": reference["optimizer"]["lr"] * lr_scale,
            "params": [
                {**group, "lr": group["lr"] * lr_scale} if "lr" in group else group
                for group in reference["optimizer"]["params"]
            ],
        },
        "train_dataloader": {
            "total_batch_size": BATCH,
            "dataset": {
                "transforms": {key: epoch(transforms[key]) for key in ("start_epoch", "mosaic_epoch", "stop_epoch")}
            },
            "collate_fn": {
                "mixup_epoch": epoch(collate["mixup_epoch"]),
                "stop_epoch": epoch(collate["stop_epoch"]),
                "copyblend_epochs": [epoch(value) for value in collate["copyblend_epochs"]],
            },
        },
        "val_dataloader": {
            "total_batch_size": reference["val_dataloader"]["total_batch_size"] // 8
        },  # 8-GPU per-GPU batch
        "lr_scale": lr_scale,
    }


def build_init(size: str, weights: Path) -> Path:
    """Repack the raw COCO detector with export_init_weights.py and drop its class-shaped head tensors."""
    import torch

    source, raw = weights / f"gtr_{size}_coco.pth", GTR / f"weights/init/gtr_{size}_coco_full_raw.pth"
    init = raw.with_name(f"gtr_{size}_coco_ul37_init.pth")
    if not init.is_file():
        subprocess.run(
            [sys.executable, "tools/export_init_weights.py", "--ckpt", source, "--out", raw], cwd=GTR, check=True
        )
        state = torch.load(raw, map_location="cpu", weights_only=True)["model"]
        torch.save({"model": {k: v for k, v in state.items() if not k.startswith(HEAD)}}, init)
    return init


def save_predictions(config: Path, checkpoint: Path, output: Path, epochs: int) -> tuple[int, int]:
    """Save final EMA detections of the val split in the canonical Ultra Benchmark JSON shape."""
    import torch
    from engine.core import YAMLConfig
    from engine.misc.dist_utils import configure_tf32_from_env

    configure_tf32_from_env()
    cfg = YAMLConfig(str(config), ViTAdapterSpatialSwiGLU={"skip_weights_warning": True})
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert state["last_epoch"] == epochs - 1, state["last_epoch"]
    model, postprocessor = cfg.model.cuda().eval(), cfg.postprocessor.cuda().eval()
    model.load_state_dict(state["ema"]["module"], strict=True)  # released checkpoints are the last epoch EMA
    ann_file = Path(cfg.yaml_cfg["val_dataloader"]["dataset"]["ann_file"])
    val_images = save_detr_predictions(model, postprocessor, cfg.val_dataloader, ann_file, output)
    torch.save({"model": state["ema"]["module"], "last_epoch": state["last_epoch"]}, output.with_name("last_ema.pth"))
    num_classes = state["ema"]["module"][
        f"decoder.dec_score_head.{cfg.yaml_cfg['GTRTransformer']['num_layers'] - 1}.bias"
    ].numel()
    return num_classes, val_images


def train_run(args: argparse.Namespace, run_dir: Path) -> None:
    """Train every selected dataset without args.yaml, then predict from its final EMA weights."""
    fraction, datasets, data_yamls, ultralytics_sha = load_roster(args.benchmark)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    import fla
    import torch
    from engine.gtr.backbone import vit_adapter

    assert vit_adapter._gla_cuda_ext is not None, "fused GLA kernel missing, run setup_gtr_env.sh"
    kernel = next((GTR / "engine/gtr/backbone/csrc").glob("gla_torch_ext*.so"))
    overrides = recipe(args.size, args.epochs)
    lr_scale = overrides.pop("lr_scale")
    source = args.weights / f"gtr_{args.size}_coco.pth"
    common = {
        "size": args.size,
        "gtr_sha": subprocess.check_output(["git", "-C", str(GTR), "rev-parse", "HEAD"], text=True).strip(),
        "gtr_patch_sha256": hashlib.sha256(
            subprocess.check_output(["git", "-C", str(GTR), "diff", "HEAD"])
        ).hexdigest(),
        "kernel_sha256": sha256(kernel),
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
        "fla_version": fla.__version__,
        "ultralytics_sha": ultralytics_sha,
        "roster_sha256": sha256(args.benchmark / "ul37_platform_uris.txt"),
        "fraction": fraction,
        "imgsz": IMGSZ,
        "batch": BATCH,
        "seed": SEED,
        "lr_scale": lr_scale,
        "source_sha256": sha256(source),
        **overrides,
    }
    pending = pending_datasets(run_dir, datasets, args.subset, common)

    init = build_init(args.size, args.weights) if pending else None
    for name, expected in pending.items():
        output_dir, scratch = run_dir / name, Path.home() / "gtr-scratch" / run_dir.parent.name / name
        lock = lock_dataset(run_dir, name)  # noqa: F841, held until the next dataset
        shutil.rmtree(output_dir, ignore_errors=True)  # no args.yaml, so anything here is from an unfinished attempt
        output_dir.mkdir(parents=True)
        scratch.mkdir(parents=True, exist_ok=True)
        num_classes = benchmark_json(args.benchmark, TO_COCO, str(data_yamls[name]), str(scratch))
        config = {
            "__include__": [str(GTR / f"configs/det/coco_finetune/gtr_{args.size}.yml")],
            "num_classes": num_classes,
            "remap_mscoco_category": False,
            "output_dir": str(scratch / "train"),
            **yaml.safe_load(yaml.safe_dump(overrides)),
        }
        for split in ("train", "val"):
            config.setdefault(f"{split}_dataloader", {}).setdefault("dataset", {}).update(
                img_folder="/", ann_file=str(scratch / f"{split}.json")
            )
        # GTR_GRAD_ACCUM_STEPS keeps 16 images per optimizer step. A rare OOM halves the micro-batch and resumes.
        accum, last = grad_accum_steps(scratch, BATCH), scratch / "train/last.pth"
        while True:
            config["train_dataloader"]["total_batch_size"] = BATCH // accum
            (scratch / "config.yml").write_text(yaml.safe_dump(config, sort_keys=False))
            print(
                f"{name}: {data_yamls[name].parent} num_classes={num_classes} "
                f"grad_accum={accum} resume={last.is_file()}",
                flush=True,
            )
            result = subprocess.run(  # train.sh: torchrun, FP32 + TF32, seed, auto-resume from output_dir without -t
                ["bash", "train.sh", *([] if last.is_file() else ["-t", str(init)])],
                cwd=GTR,
                check=False,
                env={
                    **os.environ,
                    "CONFIG": str(scratch / "config.yml"),
                    "OUTPUT_DIR": str(scratch / "train"),
                    "SEED": str(SEED),
                    "GTR_GRAD_ACCUM_STEPS": str(accum),
                    "PATH": f"{Path(sys.executable).parent}:{os.environ['PATH']}",
                },
            )
            log = max((scratch / "train").glob("train_*.log")).read_text()
            if result.returncode == 0 or accum == BATCH or "torch.OutOfMemoryError" not in log:
                result.check_returncode()
                break
            accum *= 2
            (scratch / "grad_accum").write_text(str(accum))
        check_init_load("".join(path.read_text() for path in sorted((scratch / "train").glob("train_*.log"))))
        head, val_images = save_predictions(scratch / "config.yml", last, output_dir / "predictions.json", args.epochs)
        assert head == num_classes, (head, num_classes)
        gc.collect()
        torch.cuda.empty_cache()
        for path in [scratch / "config.yml", *(scratch / "train").iterdir()]:  # logs and config, no checkpoints
            if path.suffix != ".pth" and path.name != "eval":
                shutil.move(path, output_dir / path.name)
        shutil.rmtree(scratch)
        (output_dir / "last_ema.pth").chmod(0o664)
        provenance = {
            **expected,
            "runner_sha256": RUNNER_SHA256,
            "data": str(data_yamls[name]),
            "num_classes": num_classes,
            "max_det": MAX_DET,
            "source_model": str(source),
            "init_sha256": sha256(init),
            "micro_batch": BATCH // accum,
            "grad_accum_steps": accum,
            "val_images": val_images,
            "prediction_checkpoint": str(output_dir / "last_ema.pth"),
        }
        write_args(output_dir, provenance)


def main() -> None:
    """Train or score one GTR UL37 run."""
    os.umask(0o002)
    parser = argparse.ArgumentParser()
    parser.add_argument("size", choices=tuple(SIZES))
    parser.add_argument("--device", default="0")
    parser.add_argument("--subset", help="Dataset slice like 0:5 or comma-separated dataset names")
    parser.add_argument("--name")
    parser.add_argument("--project", type=Path, required=True, help="Output root, runs go to PROJECT/NAME/multitrain")
    parser.add_argument("--benchmark", type=Path, required=True, help="UL37 benchmark checkout with val_ultrabench.py")
    parser.add_argument("--weights", type=Path, required=True, help="Dir of Phoenix8125/GTR det/gtr_<size>_coco.pth")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--val-only", action="store_true")
    parser.add_argument("--native-size-bins", action="store_true")
    args = parser.parse_args()

    run_dir = (args.project / (args.name or f"ultrabench-{PROTOCOL}-gtr-{args.size}") / "multitrain").resolve()
    if not args.val_only:
        train_run(args, run_dir)
    if args.val_only or not args.subset:
        bins = ["--native-size-bins"] if args.native_size_bins else []
        run_benchmark(args.benchmark, "val_ultrabench.py", str(run_dir), *bins)


if __name__ == "__main__":
    main()
