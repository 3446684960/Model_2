"""Paths and subprocess calls for the official comparison entry points."""

import argparse
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
DATASET = "real_world"
DATASETS = ("CAVE", "ICVL", "real_world")
DEFAULT_DATA_ROOT = Path.home() / "autodl-pub" / "Masico"
MANIFEST = json.loads((ROOT / "compare" / "official_sources.json").read_text())
MODELS = {entry["name"]: entry for entry in MANIFEST["retained_models"]}
PRIMARY_MODELS = {"EFN", "EBVIF", "UVPL"}


def model_directory(name):
    return ROOT / MODELS[name]["local_directory"]


def needs_demosaic_training(name):
    return (model_directory(name) / "train_demosaic_realworld.py").is_file()


def training_script(name, stage, dataset=DATASET):
    suffix = "realworld" if dataset == "real_world" else "simulate"
    if stage == "demosaic":
        return model_directory(name) / f"train_demosaic_{suffix}.py"
    prefix = "train" if name in PRIMARY_MODELS else "train_pansharpening"
    return model_directory(name) / f"{prefix}_{suffix}.py"


def stage_index(name, stage):
    return 1 if name in PRIMARY_MODELS or stage == "demosaic" else 2


def select_models(names):
    if names == ["all"]:
        return list(MODELS)
    if "all" in names:
        raise ValueError("Use 'all' by itself, or list individual model names.")
    return list(dict.fromkeys(names))


def best_checkpoint(work_directory, name, stage):
    directory = work_directory / str(stage_index(name, stage)) / "model"
    checkpoints = list(directory.glob("best_*.pth"))
    if not checkpoints:
        raise ValueError(f"Best checkpoint missing in {directory}")
    return max(checkpoints, key=lambda path: path.stat().st_mtime_ns)


def results_directory(work_directory, dataset=DATASET):
    return work_directory / dataset / "1" / "result" / "mat"


def check_dataset(data_root, dataset=DATASET):
    for split in ("train", "test"):
        modalities = ("mosaic", "pan") if dataset == "real_world" else ("",)
        for modality in modalities:
            directory = data_root / dataset / split / modality
            if not directory.is_dir():
                raise ValueError(f"Dataset directory missing: {directory}")
            if dataset != "real_world" and not list(directory.glob("*.mat")):
                raise ValueError(f"No MAT files in {directory}")


def dataset_defaults(dataset):
    """UVPL CAVE/real-world settings and EBVIF's published ICVL settings."""
    return {
        "epochs": 100 if dataset == "CAVE" else 10,
        "steps_per_epoch": 1000 if dataset == "real_world" else 500,
        "stride": 32 if dataset == "CAVE" else 64,
        "noise_level": [0.008, 0.01] if dataset == "ICVL" else [0.01, 0.05],
        "save_freq": {"CAVE": 10, "ICVL": 5, "real_world": 1}.get(dataset, 10),
    }


def dataset_options(args, dataset):
    options = argparse.Namespace(**vars(args))
    options.dataset = dataset
    if args.output_root is None:
        options.output_root = ROOT / "runs" / dataset
    else:
        options.output_root = args.output_root.expanduser().resolve()
        if len(args.datasets) > 1 or args.datasets == ["all"]:
            options.output_root /= dataset
    for key, value in dataset_defaults(dataset).items():
        if hasattr(options, key) and getattr(options, key) is None:
            setattr(options, key, value)
    return options


def load_official_entry(script):
    """Each official entry runs in an isolated worker process."""
    sys.path.insert(0, str(script.parent))
    specification = importlib.util.spec_from_file_location("official_entry", script)
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def run_command(command, work_directory, *, dry_run=False):
    print(shlex.join(str(argument) for argument in command), flush=True)
    if dry_run:
        return 0
    work_directory.mkdir(parents=True, exist_ok=True)
    return subprocess.run(command, cwd=work_directory, check=False).returncode
