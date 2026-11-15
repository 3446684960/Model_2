"""Shared paths and process handling for the official comparison models."""

import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
DATASET = "real_world"
DEFAULT_DATA_ROOT = Path.home() / "autodl-pub" / "Masico"
DEFAULT_OUTPUT_ROOT = ROOT / "runs" / DATASET
MANIFEST = json.loads((ROOT / "compare" / "official_sources.json").read_text())
MODELS = {entry["name"]: entry for entry in MANIFEST["retained_models"]}
PRIMARY_MODELS = {"EFN", "EBVIF", "UVPL"}


def model_directory(name):
    return ROOT / MODELS[name]["local_directory"]


def needs_demosaic_training(name):
    return (model_directory(name) / "train_demosaic_realworld.py").is_file()


def training_script(name, stage):
    if stage == "demosaic":
        return model_directory(name) / "train_demosaic_realworld.py"
    filename = "train_realworld.py" if name in PRIMARY_MODELS else "train_pansharpening_realworld.py"
    return model_directory(name) / filename


def select_models(names):
    if names == ["all"]:
        return list(MODELS)
    if "all" in names:
        raise ValueError("Use 'all' by itself, or list individual model names.")
    return list(dict.fromkeys(names))


def best_checkpoint(work_directory, stage):
    index = 1 if stage == "demosaic" else 2
    directory = work_directory / str(index) / "model"
    checkpoints = sorted(directory.glob("best_*.pth"))
    if len(checkpoints) != 1:
        raise ValueError(f"Expected one best checkpoint in {directory}, found {len(checkpoints)}.")
    return checkpoints[0]


def results_directory(work_directory):
    return work_directory / DATASET / "1" / "result" / "mat"


def check_dataset(data_root):
    for split in ("train", "test"):
        for modality in ("mosaic", "pan"):
            directory = data_root / DATASET / split / modality
            if not directory.is_dir():
                raise ValueError(f"Dataset directory missing: {directory}")


def load_official_entry(script):
    """Each entry runs in its own process so local imports stay model-specific."""
    sys.path.insert(0, str(script.parent))
    specification = importlib.util.spec_from_file_location("official_entry", script)
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def run_logged(command, work_directory, log_file):
    work_directory.mkdir(parents=True, exist_ok=True)
    print(shlex.join(str(argument) for argument in command), flush=True)
    print(f"Log file {log_file}", flush=True)
    with log_file.open("w", encoding="utf-8") as log:
        with subprocess.Popen(
            command,
            cwd=work_directory,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        ) as process:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            return process.wait()
