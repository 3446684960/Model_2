"""Shared UVPL observation simulation for the comparison datasets."""

from pathlib import Path

import h5py
import numpy as np
import scipy.io
import torch

from experiment import dataset_defaults


def noise_range(args):
    levels = getattr(args, "noise_level", None)
    if levels is None:
        levels = dataset_defaults(args.dataset)["noise_level"]
    return tuple(float(value) for value in levels)


def simulation_cache_path(args, split):
    """Separate unified observations from caches made by older readers."""
    lower, upper = noise_range(args)
    filename = (
        f"{split}_uvpl_v1_{args.dataset}_noise{lower}-{upper}"
        f"_patch{args.train_size}_stride{args.stride}"
        f"_msfa{args.msfa_size}_ratio{args.spatial_ratio}_cache.pkl"
    )
    return str(Path(args.cache_path) / filename)


def read_hyperspectral(path, dataset):
    """Read UVPL CAVE/Chikusei inputs and EFN/EBVIF ICVL inputs."""
    if dataset == "ICVL":
        with h5py.File(path, "r") as source:
            hrms = source["rad"][:]
        hrms = np.rot90(hrms.transpose(2, 1, 0))
        hrms = hrms / hrms.max((0, 1))
        return hrms[:, :, 12:28]
    if dataset == "CAVE":
        return scipy.io.loadmat(path)["b"][:, :, 12:28]
    if dataset == "chikusei":
        return scipy.io.loadmat(path)["chikusei"]
    raise ValueError(f"Unsupported simulated dataset {dataset}")


def simulate_observations(args, split):
    """Yield filename, reference HSI, low-resolution HSI, mosaic and PAN.

    The pooling, MSFA pattern, spectral response and Gaussian noise follow
    UVPL's official simulation. Sorted filenames and a split-local NumPy
    generator give every model and stage the same observations.
    """
    directory = Path(args.data_path) / args.dataset / split
    names = sorted(path.name for path in directory.iterdir() if path.is_file() and path.suffix == ".mat")
    requested = getattr(args, "data_id", [])
    # Apply selection after simulation so an image keeps its full-set noise.
    random = np.random.RandomState(42 if split == "train" else 22)
    lower, upper = noise_range(args)
    spectral_response = np.array([1., 1, 2, 4, 8, 9, 10, 12, 16, 12, 10, 9, 7, 3, 2, 1])
    spectral_response /= spectral_response.sum()
    msfa = np.arange(16).reshape(4, 4)
    kernel = np.zeros((4, 4, 16))
    for row in range(4):
        for column in range(4):
            kernel[row, column, msfa[row, column]] = 1

    for name in names:
        hrms = read_hyperspectral(directory / name, args.dataset)
        multiple = args.msfa_size * args.spatial_ratio
        height = hrms.shape[0] // multiple * multiple
        width = hrms.shape[1] // multiple * multiple
        hrms = hrms[:height, :width].astype(np.float32)

        tensor = torch.from_numpy(hrms).permute(2, 0, 1).unsqueeze(0)
        lrms = torch.nn.functional.avg_pool2d(tensor, 2, 2)
        lrms = lrms[0].permute(1, 2, 0).numpy()
        mask = np.tile(kernel, (lrms.shape[0] // 4, lrms.shape[1] // 4, 1))
        mosaic = np.sum(lrms * mask, axis=-1, keepdims=True)
        standard_deviation = random.uniform(lower, upper)
        mosaic = np.clip(mosaic + random.normal(size=mosaic.shape) * standard_deviation, 0, 1)

        pan = np.sum(hrms * spectral_response, axis=-1, keepdims=True)
        standard_deviation = random.uniform(lower, upper)
        pan = np.clip(pan + random.normal(size=pan.shape) * standard_deviation, 0, 1)

        if not requested or name in requested:
            yield name, hrms, lrms, mosaic, pan
