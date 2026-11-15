"""Generate real-world results and evaluate every model using UVPL metrics."""

import argparse
import csv
import os
from pathlib import Path
import shlex
import sys

from experiment import (
    DATASET,
    DEFAULT_DATA_ROOT,
    DEFAULT_OUTPUT_ROOT,
    MODELS,
    ROOT,
    allocate_run_id,
    best_checkpoint,
    check_dataset,
    load_official_entry,
    log_time,
    model_directory,
    needs_demosaic_training,
    resolve_run_id,
    results_directory,
    run_directory,
    run_logged,
    select_models,
)


class EvaluationRecorder:
    """Save individual and averaged validation or test scores as plain records."""

    def __init__(self, directory, model, stage):
        self.directory = directory
        self.model = model
        self.stage = stage

    def start(self, *, epoch=None, learning_rate=None, expected_images=None):
        self.epoch = epoch
        self.learning_rate = learning_rate
        self.expected_images = expected_images
        self.images = 0
        self.totals = {"QNR": 0.0, "D_lambda": 0.0, "D_S": 0.0}

    def append_csv(self, suffix, row):
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"{self.stage}_{suffix}.csv"
        write_header = not path.is_file() or path.stat().st_size == 0
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def emit(self, row, detail):
        from tqdm import tqdm

        fields = [f"time={row['time']}", f"model={self.model}", f"stage={self.stage}"]
        if self.epoch is not None:
            fields.append(f"epoch={self.epoch}")
        fields.append(detail)
        fields.extend(f"{metric}={row[metric]:.6f}" for metric in self.totals)
        line = " | ".join(fields)
        with (self.directory / f"{self.stage}.log").open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        tqdm.write(line, file=sys.stdout)

    def record(self, image, scores):
        values = {metric: float(value) for metric, value in zip(self.totals, scores)}
        self.images += 1
        for metric, value in values.items():
            self.totals[metric] += value
        row = {
            "time": log_time(), "model": self.model, "epoch": self.epoch,
            "image": image, "lr": self.learning_rate, **values,
        }
        self.append_csv("per_image", row)
        self.emit(row, f"image={image} ({self.images}/{self.expected_images})")

    def finish(self):
        if not self.images:
            raise ValueError("Evaluation has no images to summarize.")
        averages = {metric: value / self.images for metric, value in self.totals.items()}
        row = {
            "time": log_time(), "model": self.model, "epoch": self.epoch,
            "images": self.images, "lr": self.learning_rate, **averages,
        }
        self.append_csv("summary", row)
        self.emit(row, f"average_over={self.images}")


def generation_arguments(args, name, planning=False):
    work_directory = run_directory(args.output_root, name, args.run_id)
    fusion_checkpoint = "<best fusion checkpoint>"
    demosaic_checkpoint = ""
    if not planning:
        fusion_checkpoint = str(best_checkpoint(work_directory, "fusion"))
    if needs_demosaic_training(name):
        demosaic_checkpoint = "<best demosaic checkpoint>"
        if not planning:
            demosaic_checkpoint = str(best_checkpoint(work_directory, "demosaic"))
    return argparse.Namespace(
        idx=1,
        real_world=True,
        simulate=False,
        mosaic_save=True,
        pan_save=True,
        demosaic_save=True,
        gt_save=False,
        spatial_ratio=2,
        dataset=DATASET,
        num_bands=16,
        msfa_size=4,
        data_path=str(args.data_root),
        data_id=[],
        device=args.device,
        cpu=False,
        load_model=fusion_checkpoint,
        load_ps_model=fusion_checkpoint,
        load_demosaic_model=demosaic_checkpoint,
        noise_level=[0.01, 0.05],
    )


def generation_worker(args):
    name = args.models[0]
    work_directory = run_directory(args.output_root, name, args.run_id)
    work_directory.mkdir(parents=True, exist_ok=True)
    os.chdir(work_directory)
    entry = load_official_entry(model_directory(name) / "generate.py")
    entry.main(generation_arguments(args, name))


def evaluate_results(directory, device, patch_size, recorder=None):
    import numpy as np
    import scipy.io
    import torch
    import quality_index
    from tqdm import tqdm

    file_sets = {
        modality: {file.name for file in (directory / modality).glob("*.mat")}
        for modality in ("mosaic", "pan", "fused")
    }
    if not file_sets["mosaic"] or any(files != file_sets["mosaic"] for files in file_sets.values()):
        raise ValueError(f"Matching mosaic, pan and fused MAT files are required in {directory}.")
    names = sorted(file_sets["mosaic"])
    if recorder is not None:
        recorder.start(expected_images=len(names))
    msfa_kernel = torch.zeros(16, 1, 8, 8, device=device)
    totals = {"QNR": 0.0, "D_lambda": 0.0, "D_S": 0.0}
    loop = tqdm(
        names,
        desc=f"{recorder.model} Test" if recorder is not None else "Test",
        dynamic_ncols=True,
        disable=recorder is None,
        file=sys.stdout,
    )
    with torch.no_grad():
        for name in loop:
            tensors = {}
            for modality in file_sets:
                array = scipy.io.loadmat(directory / modality / name)[modality]
                tensor = torch.from_numpy(np.asarray(array, dtype=np.float64))
                tensors[modality] = tensor.permute(2, 0, 1).unsqueeze(0).to(device)
            mosaic = torch.nn.functional.pixel_shuffle(tensors["mosaic"], 4)
            scores = quality_index.calc_qnr_test(
                tensors["fused"], mosaic, tensors["pan"], msfa_kernel,
                patch_size=patch_size,
            )
            for metric, score in zip(totals, scores):
                totals[metric] += score.item()
            if recorder is not None:
                recorder.record(name, scores)
                loop.set_postfix(QNR=f"{scores[0].item():.6f}", refresh=False)
    if recorder is not None:
        recorder.finish()
    return {"images": len(names), **{metric: value / len(names) for metric, value in totals.items()}}


def save_metrics_summary(path, rows, selected):
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_name(f".{path.name}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        existing = {}
        if path.is_file():
            with path.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    row.setdefault("run_id", 0)
                    existing[(row["model"], str(row["run_id"]))] = row
        for key in selected:
            existing.pop(key, None)
        for row in rows:
            existing[(row["model"], str(row["run_id"]))] = row
        temporary = path.with_suffix(".csv.tmp")
        try:
            with temporary.open("w", newline="", encoding="utf-8") as handle:
                fields = ["model", "run_id", "images", "QNR", "D_lambda", "D_S"]
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(existing.values())
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=["all", *MODELS], default=["all"])
    parser.add_argument("--stage", choices=["all", "generate", "evaluate"], default="all")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", type=int, help="Experiment number, defaults to the latest. Use 0 for legacy outputs.")
    parser.add_argument("--device", default="0")
    parser.add_argument("--cpu", action="store_true", help="Evaluate saved results on CPU.")
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.data_root = args.data_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    if args.patch_size < 2:
        parser.error("Metric patch size must be at least 2.")
    if args.run_id is not None and args.run_id < 0:
        parser.error("Experiment number must be nonnegative.")
    try:
        names = select_models(args.models)
        if args.worker:
            if len(names) != 1 or args.stage != "generate":
                parser.error("A generation worker requires one model.")
            args.run_id = args.run_id or 0
            generation_worker(args)
            return
        if args.stage != "evaluate" and not args.dry_run:
            check_dataset(args.data_root)
    except ValueError as error:
        parser.error(str(error))

    rows, failures, selections = [], [], []
    for name in names:
        try:
            if args.dry_run and args.run_id is not None:
                run_id = args.run_id
            elif args.dry_run and not (args.output_root / name).is_dir():
                run_id = allocate_run_id(args.output_root, name, reserve=False)
            else:
                run_id = resolve_run_id(args.output_root, name, args.run_id)
        except ValueError as error:
            print(str(error), file=sys.stderr)
            failures.append(name)
            continue
        model_args = argparse.Namespace(**vars(args))
        model_args.run_id = run_id
        work_directory = run_directory(args.output_root, name, run_id)
        selections.append((name, run_id, work_directory))
        print(f"Model {name}, experiment {run_id}, output {work_directory}", flush=True)
        if args.stage in ("all", "generate"):
            command = [
                sys.executable, "-u", str(ROOT / "test.py"),
                "--worker", "--models", name, "--stage", "generate",
                "--data-root", str(args.data_root),
                "--output-root", str(args.output_root), "--device", args.device,
                "--run-id", str(run_id),
            ]
            if args.dry_run:
                print(shlex.join(command))
                print(f"Checkpoint arguments {vars(generation_arguments(model_args, name, planning=True))}")
            else:
                code = run_logged(command, work_directory, work_directory / "generate.log", model=name)
                if code:
                    failures.append(name)
                    continue
        if args.stage in ("all", "evaluate"):
            directory = results_directory(work_directory)
            if args.dry_run:
                print(f"Evaluate {name} from {directory} using UVPL QNR, patch size {args.patch_size}.")
                continue
            try:
                device = "cpu" if args.cpu else f"cuda:{args.device}"
                recorder = EvaluationRecorder(work_directory, name, "test")
                scores = evaluate_results(directory, device, args.patch_size, recorder)
                rows.append({"model": name, "run_id": run_id, **scores})
            except (ValueError, KeyError, RuntimeError, OSError) as error:
                print(f"Evaluation failed for {name}. {error}", file=sys.stderr)
                failures.append(name)
            finally:
                if not args.cpu:
                    import torch

                    torch.cuda.empty_cache()
    if not args.dry_run and args.stage in ("all", "evaluate") and selections:
        for name, run_id, work_directory in selections:
            selected = [(name, str(run_id))]
            model_rows = [row for row in rows if row["model"] == name and row["run_id"] == run_id]
            save_metrics_summary(work_directory / "metrics.csv", model_rows, selected)
            print(f"Metrics saved in {work_directory / 'metrics.csv'}")
        summary = args.output_root / "metrics.csv"
        selected = [(name, str(run_id)) for name, run_id, _ in selections]
        save_metrics_summary(summary, rows, selected)
        print(f"Metrics saved in {summary}")
    if failures:
        raise SystemExit("Failed models " + ", ".join(failures))


if __name__ == "__main__":
    main()
