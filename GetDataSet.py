import numpy
import os
import scipy.io as scio
import cv2
import torch
from torch.utils.data import Dataset
import pickle
import tqdm
from tqdm.contrib import tzip
import h5py
import scipy
import math, random


SIMULATED_PAN_RESPONSE = (1., 1, 2, 4, 8, 9, 10, 12, 16, 12, 10, 9, 7, 3, 2, 1)


def pixel_shuffle(tensor, scale_factor):
    num, channels, height, width = tensor.shape
    if channels % (scale_factor ** 2) != 0:
        raise ValueError("The channel number must be divisible by scale_factor ** 2.")

    output_channels = channels // (scale_factor ** 2)
    tensor = tensor.reshape(num, output_channels, scale_factor, scale_factor, height, width)
    tensor = tensor.transpose(0, 1, 4, 2, 5, 3)
    return tensor.reshape(num, output_channels, height * scale_factor, width * scale_factor)


def msfa_filter(data, msfa_pattern):
    height, width, num_bands = data.shape
    pattern_height, pattern_width = msfa_pattern.shape

    if num_bands != pattern_height * pattern_width:
        raise ValueError("The number of bands must equal the number of MSFA positions.")

    if height % pattern_height != 0 or width % pattern_width != 0:
        raise ValueError("The image size must be divisible by the MSFA pattern size.")

    mosaic = numpy.zeros((height, width, 1), dtype=data.dtype)
    for row in range(pattern_height):
        for col in range(pattern_width):
            band = int(msfa_pattern[row, col])
            mosaic[row::pattern_height, col::pattern_width, 0] = data[row::pattern_height, col::pattern_width, band]

    return mosaic

def crop_to_patch(img, size, stride):
    H, W = img.shape[:2]
    patches = []
    for h in range(0, H, stride):
        for w in range(0, W, stride):
            if h + size <= H and w + size <= W:
                patch = img[h: h + size, w: w + size, :]
                patches.append(patch)
    return patches

def crop_to_patch_4d(img, size, stride):
    H, W = img.shape[1:3]
    patches = []
    for h in range(0, H, stride):
        for w in range(0, W, stride):
            if h + size <= H and w + size <= W:
                patch = img[:, h: h + size, w: w + size, :]
                for f in range(patch.shape[0]):
                    patches.append(patch[f])
    return patches


def resolve_split_path(data_path, dataset, split):
    dataset_path = os.path.join(data_path, dataset)
    candidates = [
        os.path.join(dataset_path, split),
        os.path.join(dataset_path, split.capitalize()),
        os.path.join(dataset_path, f"{dataset.lower()}_{split}"),
    ]

    for path in candidates:
        if os.path.isdir(path):
            return path

    raise FileNotFoundError(f"Cannot find the {split} split. Checked: {candidates}")

class MakeSimulateDataset(Dataset):
    def __init__(self, args, type="train"):
        cache_path = os.path.join(args.cache_path, type + "_cache.pkl")
        if not os.path.exists(cache_path):
            self.mosaic, self.pan, self.hrms = [], [], []
            base_path = resolve_split_path(args.data_path, args.dataset, type)
            print("Cache file not found. Generate it from: ", base_path)
            hrms_imgs = sorted(os.listdir(base_path))

            if type == "train":
                numpy.random.seed(42)
            elif type == "test":
                numpy.random.seed(22)
            for hrms_name in tqdm.tqdm(hrms_imgs):
                if args.dataset.lower() == "cave":
                    hrms = scio.loadmat(os.path.join(base_path, hrms_name))["b"]
                    hrms = hrms[:, :, 12:28]
                elif args.dataset.lower() == "chikusei":
                    hrms = scio.loadmat(os.path.join(base_path, hrms_name))["chikusei"]
                else:
                    raise ValueError(f"Unsupported simulated dataset: {args.dataset}")

                hrms = hrms[
                    :hrms.shape[0]//(args.msfa_size*args.spatial_ratio) * (args.msfa_size*args.spatial_ratio),
                    :hrms.shape[1]//(args.msfa_size*args.spatial_ratio) * (args.msfa_size*args.spatial_ratio)
                ].astype(numpy.float32)

                MSFA = numpy.array([[0, 1, 2, 3],
                                    [4, 5, 6, 7],
                                    [8, 9, 10, 11],
                                    [12, 13, 14, 15]])

                # MS simulate
                # downsampling
                ms_tensor = torch.from_numpy(hrms).permute(2, 0, 1).unsqueeze(0)
                lrms_tensor = torch.nn.functional.avg_pool2d(ms_tensor, args.spatial_ratio, args.spatial_ratio)
                lrms = lrms_tensor[0].permute(1, 2, 0).numpy()

                # mosaicing
                mosaic = msfa_filter(lrms, MSFA)

                # adding noise, numpy.random.seed(42)
                std = numpy.random.uniform(args.noise_level[0], args.noise_level[1])
                mosaic_noise = mosaic + numpy.random.normal(size=mosaic.shape) * std
                mosaic_noise = numpy.clip(mosaic_noise, a_min=0, a_max=1.0)

                # PAN simulate
                spe_res = numpy.array(SIMULATED_PAN_RESPONSE)
                spe_res /= spe_res.sum()
                pan = numpy.sum(hrms * spe_res, axis=-1, keepdims=True)
                # adding noise, set numpy.random.seed(42)
                std = numpy.random.uniform(args.noise_level[0], args.noise_level[1])
                pan_noise = pan + numpy.random.normal(size=pan.shape) * std
                pan_noise = numpy.clip(pan_noise, a_min=0, a_max=1.0)

                spatial_ratio = pan.shape[0] // mosaic.shape[0]   #这个不就是2吗？
                if type == "train":
                    mosaic_patches = crop_to_patch(
                        mosaic_noise,
                        args.train_size//spatial_ratio,
                        args.stride//spatial_ratio
                    )
                    pan_patches = crop_to_patch(
                        pan_noise,
                        args.train_size,
                        args.stride
                    )

                    self.mosaic += mosaic_patches
                    self.pan += pan_patches

                elif type == "test":
                    self.mosaic.append(mosaic_noise)
                    self.pan.append(pan_noise)
                    self.hrms.append(hrms)

            with open(cache_path, "wb") as f:
                pickle.dump([self.mosaic, self.pan, self.hrms], f)

        print("Load data from cache file: ", cache_path)
        with open(cache_path, "rb") as f:
            self.mosaic, self.pan, self.hrms = pickle.load(f)

    def __len__(self):
        return len(self.mosaic)

    def __getitem__(self, index):
        if self.hrms != []:
            mosaic = torch.from_numpy(self.mosaic[index].astype(numpy.float32))
            mosaic = mosaic.permute(2, 0, 1)
            pan = torch.from_numpy(self.pan[index].astype(numpy.float32))
            pan = pan.permute(2, 0, 1)
            hrms = torch.from_numpy(self.hrms[index].astype(numpy.float32))
            hrms = hrms.permute(2, 0, 1)

            return mosaic, pan, hrms
        else:
            mosaic = torch.from_numpy(self.mosaic[index].astype(numpy.float32))
            mosaic = mosaic.permute(2, 0, 1)
            pan = torch.from_numpy(self.pan[index].astype(numpy.float32))
            pan = pan.permute(2, 0, 1)

            return mosaic, pan

class MakeRealWorldDataset(Dataset):
    def __init__(self, args, type="train"):
        cache_path = os.path.join(args.cache_path, type + "_cache.pkl")
        if not os.path.exists(cache_path):
            self.mosaic, self.pan = [], []
            base_path = os.path.join(args.data_path, args.dataset, type)
            print("Cache file not found. Generate it from: ", base_path)
            mosaic_path = os.path.join(base_path, "mosaic")
            pan_path = os.path.join(base_path, "pan")
            mosaic_imgs = os.listdir(mosaic_path)
            mosaic_imgs.sort()
            pan_imgs = os.listdir(pan_path)
            pan_imgs.sort()
            assert len(mosaic_imgs) == len(pan_imgs), "Length mismatch between MS and PAN images!"

            for mosaic_name, pan_name in tqdm.tqdm(tzip(mosaic_imgs, pan_imgs)):
                if args.dataset == "Ours_mat":
                    mosaic = h5py.File(os.path.join(mosaic_path, mosaic_name))["msi"][:][numpy.newaxis, :, :, :]
                    mosaic = pixel_shuffle(mosaic, args.msfa_size)[0].transpose(1, 2, 0)
                    pan = h5py.File(os.path.join(pan_path, pan_name))["pan"][:].transpose(1, 2, 0)
                elif args.dataset == "real_world":
                    with open(os.path.join(mosaic_path, mosaic_name), "rb") as f:
                        mosaic_raw = numpy.frombuffer(f.read(), dtype=numpy.int16)
                        mosaic = numpy.zeros((255*args.msfa_size, 276*args.msfa_size, 1))
                        for i in range(args.msfa_size):
                            for j in range(args.msfa_size):
                                mosaic[i::args.msfa_size, j::args.msfa_size, 0] = mosaic_raw[255*276*(i*args.msfa_size+j): 255*276*(i*args.msfa_size+j+1)].reshape(255, 276)
                    with open(os.path.join(pan_path, pan_name), "rb") as f:
                        pan_raw = numpy.frombuffer(f.read(), dtype=numpy.int16)
                        pan = pan_raw.reshape((2040, 2208, 1))
                assert pan.shape[1] % mosaic.shape[1] == 0 and pan.shape[2] % mosaic.shape[
                    2] == 0, "Mismatch between the spatial size of MS and PAN"
                spatial_ratio = pan.shape[1] // mosaic.shape[1]

                mosaic = mosaic / 2**12
                pan = pan / 2**12

                if type == "train":
                    mosaic_patches = crop_to_patch(mosaic, args.train_size // spatial_ratio, args.stride // spatial_ratio)
                    pan_patches = crop_to_patch(pan, args.train_size, args.stride)

                    self.mosaic += mosaic_patches
                    self.pan += pan_patches

                elif type == "test":
                    self.mosaic.append(mosaic)
                    self.pan.append(pan)

            with open(cache_path, "wb") as f:
                pickle.dump([self.mosaic, self.pan], f)

        print("Load data from cache file: ", cache_path)
        with open(cache_path, "rb") as f:
            self.mosaic, self.pan = pickle.load(f)

    def __len__(self):
        return len(self.mosaic)

    def __getitem__(self, index):
        mosaic = torch.from_numpy(self.mosaic[index].astype(numpy.float32))
        mosaic = mosaic.permute(2, 0, 1)
        pan = torch.from_numpy(self.pan[index].astype(numpy.float32))
        pan = pan.permute(2, 0, 1)

        return mosaic, pan

class MakeRealWorldVideoDataset(Dataset):
    def __init__(self, args, type="train"):
        cache_path = os.path.join(args.cache_path, type + "_cache.pkl")
        if not os.path.exists(cache_path):
            self.mosaic, self.pan = [], []
            base_path = os.path.join(args.data_path, args.dataset, type)
            print("Cache file not found. Generate it from: ", base_path)
            mosaic_path = os.path.join(base_path, "mosaic")
            pan_path = os.path.join(base_path, "pan")
            mosaic_imgs = os.listdir(mosaic_path)
            mosaic_imgs.sort()
            pan_imgs = os.listdir(pan_path)
            pan_imgs.sort()
            assert len(mosaic_imgs) == len(pan_imgs), "Length mismatch between MS and PAN images!"

            for mosaic_name, pan_name in tqdm.tqdm(tzip(mosaic_imgs, pan_imgs)):
                with open(os.path.join(mosaic_path, mosaic_name), "rb") as f:
                    mosaic_raw = numpy.frombuffer(f.read(), dtype=numpy.int16)
                with open(os.path.join(pan_path, pan_name), "rb") as f:
                    pan_raw = numpy.frombuffer(f.read(), dtype=numpy.int16)
                total_frames = len(mosaic_raw) // 255 // 276 // args.msfa_size**2
                mosaic = numpy.zeros((total_frames//args.frames, 255*args.msfa_size, 276*args.msfa_size, 1))
                pan = numpy.zeros((total_frames//args.frames, 255*args.msfa_size*2, 276*args.msfa_size*2, 1))
                for f in range(0, total_frames//args.frames*args.frames, args.frames):
                    for i in range(args.msfa_size):
                        for j in range(args.msfa_size):
                            mosaic[f//args.frames, i::args.msfa_size, j::args.msfa_size, 0] = mosaic_raw[f*255*276*args.msfa_size**2+255*276*(i*args.msfa_size+j): f*255*276*args.msfa_size**2+255*276*(i*args.msfa_size+j+1)].reshape(255, 276)
                    pan[f//args.frames] = pan_raw[f*255*276*args.msfa_size**2*4:(f+1)*255*276*args.msfa_size**2*4].reshape((2040, 2208, 1))

                spatial_ratio = pan.shape[1] // mosaic.shape[1]

                mosaic = mosaic / 2**12
                pan = pan / 2**12

                if type == "train":
                    mosaic_patches = crop_to_patch_4d(mosaic, args.train_size // spatial_ratio, args.stride // spatial_ratio)
                    pan_patches = crop_to_patch_4d(pan, args.train_size, args.stride)

                    self.mosaic += mosaic_patches
                    self.pan += pan_patches

                elif type == "test":
                    for f in range(mosaic.shape[0]):
                        self.mosaic.append(mosaic[f])
                        self.pan.append(pan[f])

            with open(cache_path, "wb") as f:
                pickle.dump([self.mosaic, self.pan], f)

        print("Load data from cache file: ", cache_path)
        with open(cache_path, "rb") as f:
            self.mosaic, self.pan = pickle.load(f)

    def __len__(self):
        return len(self.mosaic)

    def __getitem__(self, index):
        mosaic = torch.from_numpy(self.mosaic[index].astype(numpy.float32))
        mosaic = mosaic.permute(2, 0, 1)
        pan = torch.from_numpy(self.pan[index].astype(numpy.float32))
        pan = pan.permute(2, 0, 1)

        return mosaic, pan
