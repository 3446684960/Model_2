"""Train the retained official models with the UVPL real-world protocol."""

import argparse
from functools import wraps
import inspect
import os
from pathlib import Path
import shlex
import subprocess
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


class TrainingDisplay:
    """Observe the official loop and main optimizer for live display only."""

    def __init__(self, name, stage, epochs, recorder=None):
        self.name = name
        self.stage = stage
        self.epochs = epochs
        self.recorder = recorder
        self.frame = None
        self.bar = None
        self.optimizer = None
        self.loss_sum = 0.0
        self.loss_count = 0

    def progress(self, iterable, *args, **kwargs):
        from tqdm import tqdm

        if not isinstance(iterable, EpochBatches):
            return tqdm(iterable, *args, **kwargs)
        self.frame = sys._getframe(1)
        epoch = self.frame.f_locals["epoch"]
        self.loss_sum = 0.0
        self.loss_count = 0
        kwargs.update(
            desc=f"{self.name} {self.stage} Epoch [{epoch}/{self.epochs}]",
            dynamic_ncols=True,
            file=sys.stdout,
        )
        self.bar = tqdm(iterable, *args, **kwargs)
        if self.recorder is not None:
            self.recorder.start(
                epoch=epoch,
                learning_rate=self.optimizer.param_groups[0]["lr"],
                expected_images=self.validation_images,
            )
        return self.bar

    def after_step(self, optimizer, step_args, step_kwargs):
        if self.frame is None or self.bar is None:
            return
        loss = self.frame.f_locals.get("loss")
        if loss is None:
            return
        self.loss_sum += loss.detach().item()
        self.loss_count += 1
        self.bar.set_postfix(
            loss=f"{self.loss_sum / self.loss_count:.8f}",
            lr=f"{optimizer.param_groups[0]['lr']:.8f}",
            refresh=False,
        )

    def record_validation(self, scores):
        if self.recorder is None:
            return
        self.recorder.record(f"{self.recorder.images + 1:04d}", scores)
        if self.recorder.images == self.validation_images:
            self.recorder.finish()

    def install(self, entry):
        original_train = entry.train

        @wraps(original_train)
        def train_with_display(*args, **kwargs):
            arguments = inspect.signature(original_train).bind(*args, **kwargs).arguments
            self.optimizer = arguments.get("optimizer")
            if self.optimizer is None:
                self.optimizer = arguments["optimizers"][0]
            self.validation_images = len(arguments["val_dataloader"])
            hook = self.optimizer.register_step_post_hook(self.after_step)
            try:
                return original_train(*args, **kwargs)
            finally:
                hook.remove()
                if self.bar is not None:
                    self.bar.close()
                self.frame = None
                self.bar = None
                self.optimizer = None

        entry.train = train_with_display
        entry.tqdm = types.SimpleNamespace(tqdm=self.progress)


def training_metrics(name, on_validation=None, patch_size=64):
    """Use the same 12-bit input scale and metric settings as final testing."""
    import quality_index as uvpl_metrics

    module = types.ModuleType("quality_index")
    for attribute in dir(uvpl_metrics):
        if not attribute.startswith("__"):
            setattr(module, attribute, getattr(uvpl_metrics, attribute))

    raw_input_models = {"EBVIF", "SpNet_plus_PanGAN", "SpNet_plus_VBPN"}

    def calc_qnr_mosaic(hrms, mosaic, pan, msfa_kernel, **official_settings):
        # The other official trainers pass normalized evaluation tensors.
        if name not in raw_input_models:
            hrms, mosaic, pan = hrms * 4096, mosaic * 4096, pan * 4096
        scores = uvpl_metrics.calc_qnr_test(
            hrms,
            mosaic,
            pan,
            msfa_kernel,
            patch_size=patch_size,
        )
        if on_validation is not None:
            on_validation(scores)
        return scores

    module.calc_qnr_mosaic = calc_qnr_mosaic
    return module


def train_worker(args):
    from torch.utils.data import DataLoader
    from test import EvaluationRecorder

    name = args.models[0]
    work_directory = args.output_root / name
    work_directory.mkdir(parents=True, exist_ok=True)
    os.chdir(work_directory)

    # Official files stay intact. Their evaluation dependency is supplied here.
    recorder = None
    if args.stage == "fusion":
        recorder = EvaluationRecorder(work_directory / "2" / "record", name, "validation")
    display = TrainingDisplay(name, args.stage, args.epochs, recorder)
    sys.modules["quality_index"] = training_metrics(name, display.record_validation, args.patch_size)
    entry = load_official_entry(training_script(name, args.stage))
    display.install(entry)

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
        "--patch-size", str(args.patch_size),
    ]


def test_command(args, name):
    return [
        sys.executable, "-u", str(ROOT / "test.py"),
        "--models", name, "--stage", "all",
        "--data-root", str(args.data_root),
        "--output-root", str(args.output_root),
        "--device", args.device,
        "--patch-size", str(args.patch_size),
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
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--train-only", action="store_true", help="Complete training stages only.")
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
    if args.patch_size < 2:
        parser.error("Metric patch size must be at least 2.")
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
        fusion_completed = False
        for stage in stages:
            command = worker_command(args, name, stage)
            if args.dry_run:
                print(shlex.join(command))
                fusion_completed = stage == "fusion"
                continue
            index = 1 if stage == "demosaic" else 2
            checkpoint_directory = work_directory / str(index) / "model"
            if list(checkpoint_directory.glob("*.pth")):
                print(f"Existing checkpoints in {checkpoint_directory}. Choose another --output-root.", file=sys.stderr)
                failures.append(f"{name}/{stage}")
                break
            code = run_logged(
                command,
                work_directory,
                work_directory / f"train_{stage}.log",
                model=name,
                epochs=args.epochs,
            )
            if code:
                failures.append(f"{name}/{stage}")
                break
            fusion_completed = stage == "fusion"
        if fusion_completed and not args.train_only:
            command = test_command(args, name)
            if args.dry_run:
                print(shlex.join(command))
            elif subprocess.run(command, check=False).returncode:
                failures.append(f"{name}/test")
    if failures:
        raise SystemExit("Failed stages " + ", ".join(failures))


if __name__ == "__main__":
    main()
