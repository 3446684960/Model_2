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
