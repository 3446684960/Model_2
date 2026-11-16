"""Call the official training entries with the agreed UVPL settings."""

import argparse
import os
from pathlib import Path
import sys
import types

from experiment import (
    DATASET,
    DATASETS,
    DEFAULT_DATA_ROOT,
    MODELS,
    ROOT,
    best_checkpoint,
    check_dataset,
    dataset_options,
    load_official_entry,
    needs_demosaic_training,
    run_command,
    select_models,
    stage_index,
    training_script,
)


class EpochBatches:
    """Apply the previously agreed common batch budget to the data loader."""

    def __init__(self, loader, steps, repeat=True):
        if len(loader) == 0:
            raise ValueError("The training dataset has no batches.")
        self.loader = loader
        self.steps = steps if repeat else min(steps, len(loader))

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


def training_metrics(name, patch_size):
    """Supply the agreed final-test QNR settings to official validation."""
    import quality_index as uvpl_metrics

    module = types.ModuleType("quality_index")
    for attribute in dir(uvpl_metrics):
        if not attribute.startswith("__"):
            setattr(module, attribute, getattr(uvpl_metrics, attribute))

    raw_input_models = {"EBVIF", "SpNet_plus_PanGAN", "SpNet_plus_VBPN"}

    def calc_qnr_mosaic(hrms, mosaic, pan, msfa_kernel, **official_settings):
        if name not in raw_input_models:
            hrms, mosaic, pan = hrms * 4096, mosaic * 4096, pan * 4096
        return uvpl_metrics.calc_qnr_test(hrms, mosaic, pan, msfa_kernel, patch_size=patch_size)

    module.calc_qnr_mosaic = calc_qnr_mosaic
    return module


def train_worker(args):
    from torch.utils.data import DataLoader

    name = args.models[0]
    work_directory = args.output_root / name
    work_directory.mkdir(parents=True, exist_ok=True)
    os.chdir(work_directory)
    sys.modules["quality_index"] = training_metrics(name, args.patch_size)
    entry = load_official_entry(training_script(name, args.stage, args.dataset))

    def data_loader(*loader_args, **loader_kwargs):
        loader = DataLoader(*loader_args, **loader_kwargs)
        if loader_kwargs.get("shuffle", False):
            return EpochBatches(loader, args.steps_per_epoch, repeat=args.dataset == "real_world")
        return loader

    entry.DataLoader = data_loader
    demosaic_checkpoint = ""
    if args.stage == "fusion" and needs_demosaic_training(name):
        demosaic_checkpoint = str(best_checkpoint(work_directory, name, "demosaic"))
    official_args = argparse.Namespace(
        idx=stage_index(name, args.stage),
        dataset=args.dataset,
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
        save_freq=args.save_freq,
        device=args.device,
        num_workers=args.num_workers,
        resume="",
        resume_demosaic=demosaic_checkpoint,
        record=True,
        visual=False,
        visual_freq=1,
        noise_level=args.noise_level,
    )
    print(f"Dataset {args.dataset}, model {name}, stage {args.stage}", flush=True)
    entry.main(official_args)


def worker_command(args, name, stage):
    return [
        sys.executable, "-u", str(ROOT / "train.py"),
        "--worker", "--models", name, "--stage", stage,
        "--datasets", args.dataset,
        "--data-root", str(args.data_root),
        "--output-root", str(args.output_root),
        "--device", args.device,
        "--epochs", str(args.epochs),
        "--steps-per-epoch", str(args.steps_per_epoch),
        "--batch-size", str(args.batch_size),
        "--lr", str(args.lr),
        "--train-size", str(args.train_size),
        "--stride", str(args.stride),
        "--save-freq", str(args.save_freq),
        "--num-workers", str(args.num_workers),
        "--patch-size", str(args.patch_size),
        "--noise-level", *map(str, args.noise_level),
    ]


def test_command(args, name):
    return [
        sys.executable, "-u", str(ROOT / "test.py"),
        "--models", name, "--datasets", args.dataset,
        "--data-root", str(args.data_root),
        "--output-root", str(args.output_root),
        "--device", args.device,
        "--patch-size", str(args.patch_size),
        "--noise-level", *map(str, args.noise_level),
    ]


def run_training(args):
    names = select_models(args.models)
    if args.worker:
        if len(names) != 1 or args.stage == "all":
            raise ValueError("A worker requires one model and one training stage.")
        train_worker(args)
        return []
    if not args.dry_run:
        check_dataset(args.data_root, args.dataset)

    failures = []
    for name in names:
        stages = ["demosaic", "fusion"] if needs_demosaic_training(name) else ["fusion"]
        if args.stage != "all":
            stages = [stage for stage in stages if stage == args.stage]
        work_directory = args.output_root / name
        for stage in stages:
            if run_command(worker_command(args, name, stage), work_directory, dry_run=args.dry_run):
                failures.append(f"{args.dataset}/{name}/{stage}")
                break
        else:
            if "fusion" in stages and not args.train_only:
                if run_command(test_command(args, name), work_directory, dry_run=args.dry_run):
                    failures.append(f"{args.dataset}/{name}/test")
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=["all", *MODELS], default=["all"])
    parser.add_argument("--datasets", nargs="+", choices=["all", *DATASETS], default=[DATASET])
    parser.add_argument("--stage", choices=["all", "demosaic", "fusion"], default="all")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--device", default="0")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--steps-per-epoch", type=int)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--train-size", type=int, default=64)
    parser.add_argument("--stride", type=int)
    parser.add_argument("--save-freq", type=int)
    parser.add_argument("--noise-level", type=float, nargs=2)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.data_root = args.data_root.expanduser().resolve()
    if args.list_models:
        for name, source in MODELS.items():
            print(f"{name}\t{source['local_directory']}")
        return
    if "all" in args.datasets and args.datasets != ["all"]:
        parser.error("Use dataset 'all' by itself.")
    datasets = DATASETS if args.datasets == ["all"] else list(dict.fromkeys(args.datasets))
    if args.worker and len(datasets) != 1:
        parser.error("A worker requires one dataset.")
    failures = []
    for dataset in datasets:
        options = dataset_options(args, dataset)
        sizes = (options.epochs, options.steps_per_epoch, options.batch_size,
                 options.train_size, options.stride, options.save_freq)
        if min(sizes) < 1 or options.lr <= 0 or options.num_workers < 0:
            parser.error("Training sizes and learning rate must be positive, worker count nonnegative.")
        if options.train_size % 8 or options.stride % 8:
            parser.error("Training patch size and stride must be multiples of 8.")
        if options.steps_per_epoch > (1000 if dataset == "real_world" else 500):
            parser.error("Batch count exceeds the official UVPL loop limit.")
        if options.patch_size < 2 or not 0 <= options.noise_level[0] <= options.noise_level[1]:
            parser.error("Invalid metric patch size or noise range.")
        try:
            failures.extend(run_training(options))
        except ValueError as error:
            parser.error(str(error))
    if failures:
        raise SystemExit("Failed stages " + ", ".join(failures))


if __name__ == "__main__":
    main()
