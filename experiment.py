"""Shared paths and process handling for the official comparison models."""

import importlib.util
import codecs
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
DATASET = "real_world"
DATASETS = ("CAVE", "ICVL", "real_world")
DEFAULT_DATA_ROOT = Path.home() / "autodl-pub" / "Masico"
DEFAULT_OUTPUT_ROOT = ROOT / "runs" / DATASET
MANIFEST = json.loads((ROOT / "compare" / "official_sources.json").read_text())
MODELS = {entry["name"]: entry for entry in MANIFEST["retained_models"]}
PRIMARY_MODELS = {"EFN", "EBVIF", "UVPL"}


def log_time():
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


class PlainLog:
    """Keep epoch records and errors while terminal progress stays live."""

    def __init__(self, handle, model, epochs=None):
        self.handle = handle
        self.model = model
        self.epochs = epochs
        self.pending = ""
        self.traceback = False

    def write(self, text):
        records = re.split(r"[\r\n]", self.pending + text)
        self.pending = records.pop()
        for record in records:
            self.write_record(record)

    def write_record(self, record):
        line = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", record).strip()
        if not line or re.search(r"\d+%\||\d+it\s*\[", line):
            return
        prefix = f"time={log_time()} | model={self.model}"
        if self.epochs is None:
            output = f"{prefix} | {line}"
        else:
            epoch = re.search(r"^Epoch:\s*(\d+)", line)
            learning_rate = re.search(r"learning rate:\s*([\d.eE+-]+)", line)
            if epoch and learning_rate:
                output = f"{prefix} | epoch={epoch[1]}/{self.epochs} | lr={learning_rate[1]}"
            elif self.traceback or line.startswith("Traceback") or "error:" in line.lower():
                self.traceback = True
                output = f"{prefix} | {line}"
            else:
                return
        self.handle.write(output + "\n")
        self.handle.flush()

    def finish(self):
        self.write_record(self.pending)
        self.pending = ""


def model_directory(name):
    return ROOT / MODELS[name]["local_directory"]


def needs_demosaic_training(name):
    return (model_directory(name) / "train_demosaic_realworld.py").is_file()


def training_script(name, stage, dataset=DATASET):
    suffix = "realworld" if dataset == "real_world" else "simulate"
    if stage == "demosaic":
        return model_directory(name) / f"train_demosaic_{suffix}.py"
    filename = f"train_{suffix}.py" if name in PRIMARY_MODELS else f"train_pansharpening_{suffix}.py"
    return model_directory(name) / filename


def select_models(names):
    if names == ["all"]:
        return list(MODELS)
    if "all" in names:
        raise ValueError("Use 'all' by itself, or list individual model names.")
    return list(dict.fromkeys(names))


def run_directory(output_root, model, run_id=0):
    directory = output_root / model
    if run_id:
        directory = directory / str(run_id)
    return directory


def numbered_run(directory):
    return (
        (directory / ".run.json").is_file()
        or (directory / "1" / "model").is_dir()
        or (directory / "2" / "model").is_dir()
    )


def allocate_run_id(output_root, model, *, reserve=True):
    directory = output_root / model
    numbers = []
    if directory.is_dir():
        numbers = [int(path.name) for path in directory.iterdir() if path.is_dir() and path.name.isdecimal()]
    run_id = max(numbers, default=0) + 1
    if not reserve:
        return run_id
    directory.mkdir(parents=True, exist_ok=True)
    while True:
        experiment = run_directory(output_root, model, run_id)
        try:
            experiment.mkdir()
        except FileExistsError:
            run_id += 1
            continue
        (experiment / ".run.json").write_text(json.dumps({"model": model, "run_id": run_id}) + "\n")
        return run_id


def resolve_run_id(output_root, model, run_id=None):
    if run_id is not None:
        directory = run_directory(output_root, model, run_id)
        if not directory.is_dir():
            raise ValueError(f"Experiment directory missing: {directory}")
        if run_id and not numbered_run(directory):
            raise ValueError(f"{directory} is a legacy stage directory. Use --run-id 0 for the legacy experiment.")
        return run_id
    directory = output_root / model
    if not directory.is_dir():
        raise ValueError(f"Model output directory missing: {directory}")
    numbers = [
        int(path.name) for path in directory.iterdir()
        if path.is_dir() and path.name.isdecimal() and numbered_run(path)
    ]
    return max(numbers, default=0)


def best_checkpoint(work_directory, stage):
    index = 1 if stage == "demosaic" else 2
    directory = work_directory / str(index) / "model"
    checkpoints = sorted(directory.glob("best_*.pth"))
    if len(checkpoints) != 1:
        raise ValueError(f"Expected one best checkpoint in {directory}, found {len(checkpoints)}.")
    return checkpoints[0]


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
    }


def dataset_options(args, dataset):
    """Resolve a separate output tree and defaults for one dataset."""
    import argparse

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
    """Each entry runs in its own process so local imports stay model-specific."""
    sys.path.insert(0, str(script.parent))
    specification = importlib.util.spec_from_file_location("official_entry", script)
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def run_logged(command, work_directory, log_file, *, model=None, epochs=None):
    work_directory.mkdir(parents=True, exist_ok=True)
    print(shlex.join(str(argument) for argument in command), flush=True)
    print(f"Log file {log_file}", flush=True)
    terminal = shutil.get_terminal_size((120, 30))
    environment = dict(os.environ, COLUMNS=str(terminal.columns), LINES=str(terminal.lines))
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    with log_file.open("w", encoding="utf-8") as log:
        plain_log = PlainLog(log, model or work_directory.name, epochs)
        with subprocess.Popen(
            command,
            cwd=work_directory,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=environment,
            bufsize=0,
        ) as process:
            chunk = process.stdout.read(4096)
            while chunk:
                text = decoder.decode(chunk)
                sys.stdout.write(text)
                sys.stdout.flush()
                plain_log.write(text)
                chunk = process.stdout.read(4096)
            remaining = decoder.decode(b"", final=True)
            sys.stdout.write(remaining)
            sys.stdout.flush()
            plain_log.write(remaining)
            plain_log.finish()
            code = process.wait()
            if code:
                log.write(f"time={log_time()} | model={plain_log.model} | exit_code={code}\n")
            return code
