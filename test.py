"""Run official generation and UVPL's official test entry."""

import argparse
import os
from pathlib import Path
import sys

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
    model_directory,
    needs_demosaic_training,
    results_directory,
    run_command,
    select_models,
)


def generation_arguments(args, name):
    work_directory = args.output_root / name
    fusion_checkpoint = str(best_checkpoint(work_directory, name, "fusion"))
    demosaic_checkpoint = ""
    if needs_demosaic_training(name):
        demosaic_checkpoint = str(best_checkpoint(work_directory, name, "demosaic"))
    return argparse.Namespace(
        idx=1,
        real_world=args.dataset == "real_world",
        simulate=args.dataset != "real_world",
        mosaic_save=True,
        pan_save=True,
        demosaic_save=True,
        gt_save=args.dataset != "real_world",
        spatial_ratio=2,
        dataset=args.dataset,
        num_bands=16,
        msfa_size=4,
        data_path=str(args.data_root),
        data_id=[],
        device=args.device,
        cpu=False,
        load_model=fusion_checkpoint,
        load_ps_model=fusion_checkpoint,
        load_demosaic_model=demosaic_checkpoint,
        noise_level=args.noise_level,
    )


def test_worker(args):
    name = args.models[0]
    work_directory = args.output_root / name
    work_directory.mkdir(parents=True, exist_ok=True)
    os.chdir(work_directory)
    if args.stage == "generate":
        entry = load_official_entry(model_directory(name) / "generate.py")
        entry.main(generation_arguments(args, name))
    else:
        import quality_index

        sys.modules["quality_index"] = quality_index
        entry = load_official_entry(model_directory("UVPL") / "test.py")
        official_args = argparse.Namespace(
            real_world=args.dataset == "real_world",
            simulate=args.dataset != "real_world",
            data_path=str(results_directory(work_directory, args.dataset)),
            data_id=[],
            patch_size=args.patch_size,
            device=args.device,
            cpu=args.cpu,
        )
        entry.main(official_args)


def worker_command(args, name, stage):
    command = [
        sys.executable, "-u", str(ROOT / "test.py"),
        "--worker", "--models", name, "--stage", stage,
        "--datasets", args.dataset,
        "--data-root", str(args.data_root),
        "--output-root", str(args.output_root),
        "--device", args.device,
        "--patch-size", str(args.patch_size),
        "--noise-level", *map(str, args.noise_level),
    ]
    if args.cpu:
        command.append("--cpu")
    return command


def run_testing(args):
    names = select_models(args.models)
    if args.worker:
        if len(names) != 1 or args.stage == "all":
            raise ValueError("A worker requires one model and one test stage.")
        test_worker(args)
        return []
    if args.stage != "evaluate" and not args.dry_run:
        check_dataset(args.data_root, args.dataset)

    failures = []
    stages = ["generate", "evaluate"] if args.stage == "all" else [args.stage]
    for name in names:
        print(f"Dataset {args.dataset}, model {name}", flush=True)
        for stage in stages:
            if run_command(worker_command(args, name, stage), args.output_root / name, dry_run=args.dry_run):
                failures.append(f"{args.dataset}/{name}/{stage}")
                break
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=["all", *MODELS], default=["all"])
    parser.add_argument("--datasets", nargs="+", choices=["all", *DATASETS], default=[DATASET])
    parser.add_argument("--stage", choices=["all", "generate", "evaluate"], default="all")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--device", default="0")
    parser.add_argument("--cpu", action="store_true", help="Evaluate saved results on CPU.")
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--noise-level", type=float, nargs=2)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.data_root = args.data_root.expanduser().resolve()
    if "all" in args.datasets and args.datasets != ["all"]:
        parser.error("Use dataset 'all' by itself.")
    datasets = DATASETS if args.datasets == ["all"] else list(dict.fromkeys(args.datasets))
    if args.worker and len(datasets) != 1:
        parser.error("A worker requires one dataset.")
    failures = []
    for dataset in datasets:
        options = dataset_options(args, dataset)
        if options.patch_size < 2 or not 0 <= options.noise_level[0] <= options.noise_level[1]:
            parser.error("Invalid metric patch size or noise range.")
        try:
            failures.extend(run_testing(options))
        except ValueError as error:
            parser.error(str(error))
    if failures:
        raise SystemExit("Failed stages " + ", ".join(failures))


if __name__ == "__main__":
    main()
