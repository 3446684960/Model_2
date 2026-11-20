import argparse
import json
import math
import os
import time

import torch
import torch.nn as nn
import tqdm
from torch.utils.data import DataLoader

import quality_index
from GetDataSet import MakeSimulateDataset, SIMULATED_PAN_RESPONSE
from degradation import MosaicDegradation, create_wald_inputs
from model import FusionNet


def clipped_gaussian_nll(prediction, observation, sigma):
    interior = 0.5 * ((observation - prediction) / sigma).square() + math.log(sigma * math.sqrt(2 * math.pi))
    lower_tail = -torch.special.log_ndtr(-prediction / sigma)
    upper_tail = -torch.special.log_ndtr((prediction - 1) / sigma)
    return torch.where(observation <= 0, lower_tail, torch.where(observation >= 1, upper_tail, interior)).mean()


def wald_measurements(fuse_net, mosaic_degradation, mosaic, pan, args, generator=None):
    mosaic_low, pan_low, holdout = create_wald_inputs(mosaic, pan, args.msfa_size, args.spatial_ratio, generator)
    wald_hsi = fuse_net(mosaic_low, pan_low)
    mosaic_hat = mosaic_degradation.sample_msfa(wald_hsi)
    return mosaic_hat[holdout], mosaic[holdout]


def train(fuse_net: nn.Module, mosaic_degradation: nn.Module,
          optimizer, train_dataloader, val_dataloader, args):
    print("===> Begin Training!")

    device = next(fuse_net.parameters()).device
    start_epoch = getattr(args, "start_epoch", 0)
    best_epoch = getattr(args, "best_epoch", 0)
    best_psnr = getattr(args, "best_psnr", float("-inf"))
    total_start_time = time.time()

    for epoch in range(start_epoch + 1, args.epochs + 1):
        fuse_net.train()
        epoch_start_time = time.time()

        loss_sum = 0.0

        for mosaic, pan in tqdm.tqdm(train_dataloader, desc=f"Epoch {epoch}/{args.epochs}"):
            mosaic = mosaic.to(device, non_blocking=True)
            pan = pan.to(device, non_blocking=True)

            optimizer.zero_grad()

            prediction, observation = wald_measurements(fuse_net, mosaic_degradation, mosaic, pan, args)
            loss = clipped_gaussian_nll(prediction, observation, args.mosaic_sigma)

            loss.backward()
            optimizer.step()

            loss_sum += loss.detach().item()

        loss_avg = loss_sum / len(train_dataloader)

        fuse_net.eval()
        psnr_sum = 0.0
        sam_sum = 0.0
        wald_val_error = 0.0
        wald_val_count = 0
        val_generator = torch.Generator(device=device).manual_seed(0)
        ratio = args.spatial_ratio
        alignment = args.msfa_size * ratio

        with torch.no_grad():
            for mosaic, pan, hrhsi in val_dataloader:
                mosaic = mosaic.to(device, non_blocking=True)
                pan = pan.to(device, non_blocking=True)
                hrhsi = hrhsi.to(device, non_blocking=True)

                prediction = fuse_net(mosaic, pan)
                psnr_sum += quality_index.calc_psnr(hrhsi, prediction).item()
                sam_sum += quality_index.calc_sam(hrhsi, prediction).item()

                h = mosaic.shape[-2] // alignment * alignment
                w = mosaic.shape[-1] // alignment * alignment
                if min(h, w) >= 2 * alignment:
                    val_prediction, val_observation = wald_measurements(
                        fuse_net, mosaic_degradation, mosaic[..., :h, :w], pan[..., :h * ratio, :w * ratio],
                        args, val_generator
                    )
                    wald_val_error += (val_prediction - val_observation).abs().sum().item()
                    wald_val_count += val_observation.numel()

        psnr_avg = psnr_sum / len(val_dataloader)
        sam_avg = sam_sum / len(val_dataloader)
        wald_val_mae = wald_val_error / wald_val_count if wald_val_count else None
        checkpoint = {
            "training_mode": "wald_ea",
            "epoch": epoch,
            "fuse_net": fuse_net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_psnr": max(best_psnr, psnr_avg),
            "best_epoch": epoch if psnr_avg > best_psnr else best_epoch,
        }

        if psnr_avg > best_psnr:
            best_psnr = psnr_avg
            best_epoch = epoch
            torch.save(checkpoint, os.path.join(args.dir_model, "best.pth"))

        if epoch % args.save_freq == 0:
            torch.save(checkpoint, os.path.join(args.dir_model, f"epoch_{epoch}.pth"))

        if args.record:
            record = []
            if os.path.exists(args.record_path):
                with open(args.record_path, "r") as file:
                    record = json.load(file)

            record.append({
                "epoch": epoch,
                "loss": loss_avg,
                "wald_val_mae": wald_val_mae,
                "psnr": psnr_avg,
                "sam": sam_avg,
                "best_psnr": best_psnr,
                "best_epoch": best_epoch,
            })

            with open(args.record_path, "w") as file:
                json.dump(record, file, indent=2)

        val_mae = f"{wald_val_mae:.6f}" if wald_val_mae is not None else "n/a"
        print(f"Epoch: {epoch} | wald NLL: {loss_avg:.6f} | val Wald MAE: {val_mae} | psnr: {psnr_avg:.4f} | "
              f"sam: {sam_avg:.4f} | best: {best_psnr:.4f} ({best_epoch}) | "
              f"time: {(time.time() - epoch_start_time) / 60:.2f} min")

    print(f"Total time: {(time.time() - total_start_time) / 60:.2f} min")
    print(f"Best epoch: {best_epoch}, Best PSNR: {best_psnr:.4f}")


def create_experiment_dir(args):
    version_dir = os.path.join(args.output_root, args.dataset, args.model_name, args.version_name)
    os.makedirs(version_dir, exist_ok=True)

    run_ids = [int(name) for name in os.listdir(version_dir)
               if name.isdigit() and os.path.isdir(os.path.join(version_dir, name))]
    run_id = max(run_ids, default=-1) + 1
    experiment_dir = os.path.join(version_dir, str(run_id))
    os.makedirs(experiment_dir)

    print("训练文件夹为：{}".format(experiment_dir))
    return experiment_dir


def main(args):
    if args.msfa_size != 4 or args.num_bands != len(SIMULATED_PAN_RESPONSE):
        raise ValueError("The simulator uses a 4x4 MSFA with 16 bands.")
    if args.spatial_ratio < 2:
        raise ValueError("Wald training requires spatial_ratio >= 2.")
    msfa_alignment = args.msfa_size * args.spatial_ratio
    wald_alignment = msfa_alignment * args.spatial_ratio
    if args.train_size < 2 * wald_alignment or args.train_size % wald_alignment:
        raise ValueError(f"train_size must be a multiple of {wald_alignment} and at least {2 * wald_alignment}.")
    if args.stride < 1 or args.stride % msfa_alignment:
        raise ValueError(f"stride must be a positive multiple of {msfa_alignment} to preserve MSFA phase.")
    if len(args.noise_level) != 2 or not 0 <= args.noise_level[0] <= args.noise_level[1]:
        raise ValueError("noise_level must contain two non-negative bounds in ascending order.")
    default_sigma = sum(args.noise_level) / 2
    args.mosaic_sigma = default_sigma if args.mosaic_sigma is None else args.mosaic_sigma
    if not math.isfinite(args.mosaic_sigma) or args.mosaic_sigma <= 0:
        raise ValueError("mosaic_sigma must be positive and finite.")

    experiment_dir = create_experiment_dir(args)
    args.cache_path = experiment_dir
    args.dir_model = os.path.join(experiment_dir, "model")
    args.record_path = os.path.join(experiment_dir, "record.json")

    os.makedirs(args.dir_model, exist_ok=True)

    use_cuda = torch.cuda.is_available()
    device = torch.device(f"cuda:{args.device}" if use_cuda else "cpu")
    print("Device:", device)

    train_set = MakeSimulateDataset(args, "train")
    val_set = MakeSimulateDataset(args, "test")

    train_dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=use_cuda)
    val_dataloader = DataLoader(val_set, batch_size=1, shuffle=False,
                                num_workers=args.num_workers, pin_memory=use_cuda)

    fuse_net = FusionNet(num_bands=args.num_bands, num_endmembers=args.num_endmembers, msfa_size=args.msfa_size,
                         spatial_ratio=args.spatial_ratio, feature_channels=args.feature_channels).to(device)
    mosaic_degradation = MosaicDegradation(num_bands=args.num_bands, msfa_size=args.msfa_size,
                                           spatial_ratio=args.spatial_ratio).to(device)
    optimizer = torch.optim.Adam(fuse_net.parameters(), lr=args.lr)

    args.start_epoch = 0
    args.best_epoch = 0
    args.best_psnr = float("-inf")

    if args.resume:
        checkpoint_path = os.path.expanduser(args.resume)
        print("==> Load checkpoint:", checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if checkpoint.get("training_mode") != "wald_ea":
            raise ValueError("--resume requires a Wald EA checkpoint.")
        fuse_net.load_state_dict(checkpoint["fuse_net"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        args.start_epoch = checkpoint.get("epoch", 0)
        args.best_epoch = checkpoint.get("best_epoch", args.start_epoch)
        args.best_psnr = checkpoint.get("best_psnr", float("-inf"))
    else:
        print("==> Train from scratch")

    train(fuse_net, mosaic_degradation, optimizer, train_dataloader, val_dataloader, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Wald Mosaic–PAN fusion with EA")
    parser.add_argument("--dataset", type=str, default="CAVE")
    parser.add_argument("--output_root", type=str, default=".")
    parser.add_argument("--model_name", type=str, default="Model_L_2")
    parser.add_argument("--version_name", type=str, default="v0")
    parser.add_argument("--train_size", type=int, default=128)
    parser.add_argument("--msfa_size", type=int, default=4)
    parser.add_argument("--noise_level", nargs="+", type=float, default=[0.01, 0.05])
    parser.add_argument("--spatial_ratio", type=int, default=2)
    parser.add_argument("--num_bands", type=int, default=16)
    parser.add_argument("--num_endmembers", type=int, default=6)
    parser.add_argument("--feature_channels", type=int, default=64)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--save_freq", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--mosaic_sigma", type=float, default=None, help="Mosaic observation noise, defaults to midpoint of noise_level")
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--data_path", type=str, default=os.path.expanduser("~/autodl-pub/Masico"))
    parser.add_argument("--record", action=argparse.BooleanOptionalAction, default=True)

    main(parser.parse_args())
