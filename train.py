"""Train the retained official models with the UVPL real-world protocol."""

import argparse
import os
from pathlib import Path
import shlex
import sys
import types

from experiment import (
    DATASET,
    DEFAULT_DATA_ROOT,
    DEFAULT_OUTPUT_ROOT,
    MODELS,
    ROOT,
    best_checkpoint,
    check_dataset,
    load_official_entry,
    needs_demosaic_training,
    run_logged,
    select_models,
    training_script,
)


class EpochBatches:
    """Yield the same number of training batches per epoch for every model."""

    def __init__(self, loader, steps):
        if len(loader) == 0:
            raise ValueError("The training dataset has no batches.")
        self.loader = loader
        self.steps = steps

    def __len__(self):
        return self.steps

    def __iter__(self):
        iterator = iter(self.loader)
        for _ in range(self.steps):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(self.loader)
                batch = next(iterator)
            yield batch


def training_metrics(name):
    """Use UVPL validation settings and explicitly convert raw 12-bit inputs."""
    import quality_index as uvpl_metrics

    module = types.ModuleType("quality_index")
    for attribute in dir(uvpl_metrics):
        if not attribute.startswith("__"):
            setattr(module, attribute, getattr(uvpl_metrics, attribute))

    raw_input_models = {"EBVIF", "SpNet_plus_PanGAN", "SpNet_plus_VBPN"}

    def calc_qnr_mosaic(hrms, mosaic, pan, msfa_kernel, **official_settings):
        # These three official trainers pass all inputs in 12-bit integer units.
        if name in raw_input_models:
            hrms, mosaic, pan = hrms / 4096, mosaic / 4096, pan / 4096
        return uvpl_metrics.calc_qnr_mosaic(
            hrms,
            mosaic,
            pan,
            msfa_kernel,
            patch_size=128,
            scale_factor=2,
            p=0.3,
            q=1,
            alpha=0.8,
            beta=1.2,
        )

    module.calc_qnr_mosaic = calc_qnr_mosaic
    return module


def train_worker(args):
    from torch.utils.data import DataLoader

    name = args.models[0]
    work_directory = args.output_root / name
    work_directory.mkdir(parents=True, exist_ok=True)
    os.chdir(work_directory)

    # Official files stay intact. Their evaluation dependency is supplied here.
    sys.modules["quality_index"] = training_metrics(name)
    entry = load_official_entry(training_script(name, args.stage))

    def data_loader(*loader_args, **loader_kwargs):
        loader = DataLoader(*loader_args, **loader_kwargs)
        if loader_kwargs.get("shuffle", False):
            return EpochBatches(loader, args.steps_per_epoch)
        return loader

    entry.DataLoader = data_loader
    demosaic_checkpoint = ""
    if args.stage == "fusion" and needs_demosaic_training(name):
        demosaic_checkpoint = str(best_checkpoint(work_directory, "demosaic"))
    official_args = argparse.Namespace(
        idx=1 if args.stage == "demosaic" else 2,
        dataset=DATASET,
        data_path=str(args.data_root),
        train_size=args.train_size,
        stride=args.stride,
        msfa_size=4,
        spatial_ratio=2,
        num_bands=16,
        batch_size=args.batch_size,
        epochs=args.epochs,
        iters_per_epoch=args.steps_per_epoch,
        optimizer="adam",
        lr=args.lr,
        lr_decay=False,
        save_freq=1,
        device=args.device,
        num_workers=args.num_workers,
        resume="",
        resume_demosaic=demosaic_checkpoint,
        record=True,
        visual=True,
        visual_freq=1,
    )
    print(f"Model {name}, stage {args.stage}", flush=True)
    print(f"Unified training arguments {vars(official_args)}", flush=True)
    entry.main(official_args)


def worker_command(args, name, stage):
    return [
        sys.executable, "-u", str(ROOT / "train.py"),
        "--worker", "--models", name, "--stage", stage,
        "--data-root", str(args.data_root),
        "--output-root", str(args.output_root),
        "--device", args.device,
        "--epochs", str(args.epochs),
        "--steps-per-epoch", str(args.steps_per_epoch),
        "--batch-size", str(args.batch_size),
        "--lr", str(args.lr),
        "--train-size", str(args.train_size),
        "--stride", str(args.stride),
        "--num-workers", str(args.num_workers),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=["all", *MODELS], default=["all"])
    parser.add_argument("--stage", choices=["all", "demosaic", "fusion"], default="all")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", default="0")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--steps-per-epoch", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--train-size", type=int, default=64)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.data_root = args.data_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    if args.list_models:
        for name, source in MODELS.items():
            print(f"{name}\t{source['local_directory']}")
        return
    if min(args.epochs, args.steps_per_epoch, args.batch_size, args.train_size, args.stride) < 1:
        parser.error("Training sizes and iteration counts must be positive.")
    if args.steps_per_epoch > 1000:
        parser.error("The official UVPL trainer supports at most 1000 batches per epoch.")
    if args.train_size % 8 or args.stride % 8:
        parser.error("Training patch size and stride must be multiples of 8 for the filter array.")
    if args.lr <= 0 or args.num_workers < 0:
        parser.error("Learning rate must be positive and worker count must be nonnegative.")
    try:
        names = select_models(args.models)
        if args.worker:
            if len(names) != 1 or args.stage == "all":
                parser.error("A worker requires one model and one training stage.")
            train_worker(args)
            return
        if not args.dry_run:
            check_dataset(args.data_root)
    except ValueError as error:
        parser.error(str(error))

    failures = []
    for name in names:
        stages = ["demosaic", "fusion"] if needs_demosaic_training(name) else ["fusion"]
        if args.stage != "all":
            stages = [stage for stage in stages if stage == args.stage]
        work_directory = args.output_root / name
        for stage in stages:
            command = worker_command(args, name, stage)
            if args.dry_run:
                print(shlex.join(command))
                continue
            index = 1 if stage == "demosaic" else 2
            checkpoint_directory = work_directory / str(index) / "model"
            if list(checkpoint_directory.glob("*.pth")):
                print(f"Existing checkpoints in {checkpoint_directory}. Choose another --output-root.", file=sys.stderr)
                failures.append(f"{name}/{stage}")
                break
            code = run_logged(command, work_directory, work_directory / f"train_{stage}.log")
            if code:
                failures.append(f"{name}/{stage}")
                break
    if failures:
        raise SystemExit("Failed training stages " + ", ".join(failures))


if __name__ == "__main__":
    main()
