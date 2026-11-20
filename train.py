import argparse
import json
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from torch.utils.data import DataLoader

import quality_index
from GetDataSet import MakeSimulateDataset, SIMULATED_PAN_RESPONSE
from degradation import MosaicDegradation, PanDegradation, create_wald_inputs, srf_anchor_observations
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
    return mosaic_hat[holdout], mosaic[holdout], wald_hsi, pan_low


def train(fuse_net: nn.Module, mosaic_degradation: nn.Module, pan_degradation: nn.Module,
          optimizer, optimizer_srf, train_dataloader, val_dataloader, args):
    print("===> Begin Training!")

    device = next(fuse_net.parameters()).device
    start_epoch = getattr(args, "start_epoch", 0)
    best_epoch = getattr(args, "best_epoch", 0)
    best_psnr = getattr(args, "best_psnr", float("-inf"))
    total_start_time = time.time()
    ratio = args.spatial_ratio
    alignment = args.msfa_size * ratio
    pan_scale = args.pan_residual_scale
    reference_srf = torch.tensor(SIMULATED_PAN_RESPONSE)
    reference_srf /= reference_srf.sum()

    for epoch in range(start_epoch + 1, args.epochs + 1):
        fuse_net.train()
        epoch_start_time = time.time()
        use_wald_pan = epoch > args.srf_warmup_end

        loss_sum = 0.0
        loss_mosaic_sum = 0.0
        loss_pan_low_sum = 0.0
        loss_srf_anchor_sum = 0.0
        loss_srf_wald_sum = 0.0

        for mosaic, pan in tqdm.tqdm(train_dataloader, desc=f"Epoch {epoch}/{args.epochs}"):
            mosaic = mosaic.to(device, non_blocking=True)
            pan = pan.to(device, non_blocking=True)

            # 更新 EA 时，SRF 只作为固定的光谱投影。
            optimizer.zero_grad()
            prediction, observation, wald_hsi, wald_pan = wald_measurements(fuse_net, mosaic_degradation, mosaic, pan, args)
            loss_mosaic = clipped_gaussian_nll(prediction, observation, args.mosaic_sigma)
            loss_pan_low = wald_hsi.new_zeros(())
            if use_wald_pan:
                wald_spectra = F.avg_pool2d(wald_hsi, ratio, ratio)
                wald_pan_mean = F.avg_pool2d(wald_pan, ratio, ratio)
                response = pan_degradation.get_spectral_response().detach()
                pan_hat = F.conv2d(wald_spectra, response)
                loss_pan_low = F.smooth_l1_loss(pan_hat / pan_scale, wald_pan_mean / pan_scale)
            loss = loss_mosaic + args.lambda_pan_low * loss_pan_low
            loss.backward()
            optimizer.step()

            # 更新 SRF 时，用实测配对做锚点，Wald 重建只提供固定的光谱估计。
            optimizer_srf.zero_grad()
            spectra, pan_mean, weights = srf_anchor_observations(mosaic, pan, args.msfa_size, ratio)
            anchor_error = F.smooth_l1_loss(pan_degradation(spectra) / pan_scale, pan_mean / pan_scale, reduction="none")
            loss_srf_anchor = (weights * anchor_error).sum() / weights.sum().clamp_min(1e-6)
            loss_srf_wald = wald_hsi.new_zeros(())
            if use_wald_pan:
                pan_hat = pan_degradation(wald_spectra.detach())
                loss_srf_wald = F.smooth_l1_loss(pan_hat / pan_scale, wald_pan_mean / pan_scale)
            loss_srf = loss_srf_anchor + args.lambda_srf_wald * loss_srf_wald
            loss_srf.backward()
            optimizer_srf.step()

            loss_sum += loss.detach().item()
            loss_mosaic_sum += loss_mosaic.detach().item()
            loss_pan_low_sum += loss_pan_low.detach().item()
            loss_srf_anchor_sum += loss_srf_anchor.detach().item()
            loss_srf_wald_sum += loss_srf_wald.detach().item()

        train_batches = len(train_dataloader)
        loss_avg = loss_sum / train_batches
        loss_mosaic_avg = loss_mosaic_sum / train_batches
        loss_pan_low_avg = loss_pan_low_sum / train_batches
        loss_srf_anchor_avg = loss_srf_anchor_sum / train_batches
        loss_srf_wald_avg = loss_srf_wald_sum / train_batches

        fuse_net.eval()
        psnr_sum = 0.0
        sam_sum = 0.0
        wald_val_error = 0.0
        wald_val_count = 0
        srf_val_error = 0.0
        srf_val_weight = 0.0
        val_generator = torch.Generator(device=device).manual_seed(0)

        with torch.no_grad():
            for mosaic, pan, hrhsi in val_dataloader:
                mosaic = mosaic.to(device, non_blocking=True)
                pan = pan.to(device, non_blocking=True)
                hrhsi = hrhsi.to(device, non_blocking=True)

                prediction = fuse_net(mosaic, pan)
                psnr_sum += quality_index.calc_psnr(hrhsi, prediction).item()
                sam_sum += quality_index.calc_sam(hrhsi, prediction).item()

                spectra, pan_mean, weights = srf_anchor_observations(mosaic, pan, args.msfa_size, ratio)
                srf_val_error += (weights * (pan_degradation(spectra) - pan_mean).abs()).sum().item()
                srf_val_weight += weights.sum().item()

                h = mosaic.shape[-2] // alignment * alignment
                w = mosaic.shape[-1] // alignment * alignment
                if min(h, w) >= 2 * alignment:
                    val_prediction, val_observation, _, _ = wald_measurements(
                        fuse_net, mosaic_degradation, mosaic[..., :h, :w], pan[..., :h * ratio, :w * ratio],
                        args, val_generator
                    )
                    wald_val_error += (val_prediction - val_observation).abs().sum().item()
                    wald_val_count += val_observation.numel()

        psnr_avg = psnr_sum / len(val_dataloader)
        sam_avg = sam_sum / len(val_dataloader)
        wald_val_mae = wald_val_error / wald_val_count if wald_val_count else None
        srf_val_mae = srf_val_error / srf_val_weight if srf_val_weight else None
        spectral_response = pan_degradation.get_spectral_response().detach().flatten().cpu()
        srf_rmse = (spectral_response - reference_srf).square().mean().sqrt().item()

        checkpoint = {
            "training_mode": "wald_ea_srf",
            "epoch": epoch,
            "fuse_net": fuse_net.state_dict(),
            "pan_degradation": pan_degradation.state_dict(),
            "optimizer": optimizer.state_dict(),
            "optimizer_srf": optimizer_srf.state_dict(),
            "srf_warmup_end": args.srf_warmup_end,
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
                "loss_mosaic": loss_mosaic_avg,
                "loss_pan_low": loss_pan_low_avg,
                "loss_srf_anchor": loss_srf_anchor_avg,
                "loss_srf_wald": loss_srf_wald_avg,
                "wald_val_mae": wald_val_mae,
                "srf_val_mae": srf_val_mae,
                "srf_rmse": srf_rmse,
                "pan_response": spectral_response.tolist(),
                "psnr": psnr_avg,
                "sam": sam_avg,
                "best_psnr": best_psnr,
                "best_epoch": best_epoch,
            })

            with open(args.record_path, "w") as file:
                json.dump(record, file, indent=2)

        val_mae = f"{wald_val_mae:.6f}" if wald_val_mae is not None else "n/a"
        srf_mae = f"{srf_val_mae:.6f}" if srf_val_mae is not None else "n/a"
        print(f"Epoch: {epoch} | Wald NLL: {loss_mosaic_avg:.6f} | PAN low: {loss_pan_low_avg:.6f} | "
              f"val Wald MAE: {val_mae} | psnr: {psnr_avg:.4f} | "
              f"sam: {sam_avg:.4f} | best: {best_psnr:.4f} ({best_epoch}) | "
              f"time: {(time.time() - epoch_start_time) / 60:.2f} min")
        print(f"SRF anchor: {loss_srf_anchor_avg:.6f} | SRF Wald: {loss_srf_wald_avg:.6f} | "
              f"val SRF MAE: {srf_mae} | SRF RMSE: {srf_rmse:.6f}")

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
    if not all(math.isfinite(value) and value > 0 for value in (args.lr_srf, args.pan_residual_scale)):
        raise ValueError("lr_srf and pan_residual_scale must be positive and finite.")
    if not all(math.isfinite(value) and value >= 0 for value in (args.lambda_pan_low, args.lambda_srf_wald)):
        raise ValueError("PAN and SRF loss weights must be non-negative and finite.")
    if args.srf_warmup_epochs < 0:
        raise ValueError("srf_warmup_epochs must be non-negative.")

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
    pan_degradation = PanDegradation(num_bands=args.num_bands).to(device)
    optimizer = torch.optim.Adam(fuse_net.parameters(), lr=args.lr)
    optimizer_srf = torch.optim.Adam(pan_degradation.parameters(), lr=args.lr_srf)

    args.start_epoch = 0
    args.best_epoch = 0
    args.best_psnr = float("-inf")
    args.srf_warmup_end = args.srf_warmup_epochs

    if args.resume:
        checkpoint_path = os.path.expanduser(args.resume)
        print("==> Load checkpoint:", checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        training_mode = checkpoint.get("training_mode")
        if training_mode not in ("wald_ea", "wald_ea_srf"):
            raise ValueError("--resume requires a Wald EA checkpoint.")
        fuse_net.load_state_dict(checkpoint["fuse_net"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        args.start_epoch = checkpoint.get("epoch", 0)
        args.best_epoch = checkpoint.get("best_epoch", args.start_epoch)
        args.best_psnr = checkpoint.get("best_psnr", float("-inf"))
        if training_mode == "wald_ea_srf":
            pan_degradation.load_state_dict(checkpoint["pan_degradation"])
            optimizer_srf.load_state_dict(checkpoint["optimizer_srf"])
            args.srf_warmup_end = checkpoint["srf_warmup_end"]
        else:
            args.srf_warmup_end = args.start_epoch + args.srf_warmup_epochs
    else:
        print("==> Train from scratch")

    train(fuse_net, mosaic_degradation, pan_degradation, optimizer, optimizer_srf,
          train_dataloader, val_dataloader, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Wald Mosaic–PAN fusion with EA and learn PAN SRF")
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
    parser.add_argument("--lr_srf", type=float, default=1e-3)
    parser.add_argument("--srf_warmup_epochs", type=int, default=10)
    parser.add_argument("--lambda_pan_low", type=float, default=0.1)
    parser.add_argument("--lambda_srf_wald", type=float, default=0.1)
    parser.add_argument("--pan_residual_scale", type=float, default=0.03)
    parser.add_argument("--mosaic_sigma", type=float, default=None, help="Mosaic observation noise, defaults to midpoint of noise_level")
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--data_path", type=str, default=os.path.expanduser("~/autodl-pub/Masico"))
    parser.add_argument("--record", action=argparse.BooleanOptionalAction, default=True)

    main(parser.parse_args())
