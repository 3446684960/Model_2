import argparse
import json
import os
import time

import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from torch.utils.data import DataLoader

import quality_index
from GetDataSet import MakeSimulateDataset
from degradation import MosaicDegradation, PanDegradation
from model import FusionNet


def train(fuse_net: nn.Module, mosaic_degradation: nn.Module, pan_degradation: nn.Module,
          optimizer, train_dataloader, val_dataloader, args):
    print("===> Begin Training!")

    device = next(fuse_net.parameters()).device
    start_epoch = getattr(args, "start_epoch", 0)
    best_epoch = getattr(args, "best_epoch", 0)
    best_psnr = getattr(args, "best_psnr", float("-inf"))
    total_start_time = time.time()

    for epoch in range(start_epoch + 1, args.epochs + 1):
        fuse_net.train()
        pan_degradation.train()
        epoch_start_time = time.time()

        loss_sum = 0.0
        loss_mosaic_sum = 0.0
        loss_pan_sum = 0.0

        for mosaic, pan in tqdm.tqdm(train_dataloader, desc=f"Epoch {epoch}/{args.epochs}"):
            mosaic = mosaic.to(device, non_blocking=True)
            pan = pan.to(device, non_blocking=True)

            optimizer.zero_grad()

            hrhsi_hat = fuse_net(mosaic, pan)
            mosaic_hat = mosaic_degradation(hrhsi_hat)
            pan_hat = pan_degradation(hrhsi_hat)

            loss_mosaic = F.l1_loss(mosaic_hat, mosaic)
            loss_pan = F.l1_loss(pan_hat, pan)
            loss = loss_mosaic + args.lambda_pan * loss_pan

            loss.backward()
            optimizer.step()

            loss_sum += loss.detach().item()
            loss_mosaic_sum += loss_mosaic.detach().item()
            loss_pan_sum += loss_pan.detach().item()

        loss_avg = loss_sum / len(train_dataloader)
        loss_mosaic_avg = loss_mosaic_sum / len(train_dataloader)
        loss_pan_avg = loss_pan_sum / len(train_dataloader)

        fuse_net.eval()
        pan_degradation.eval()
        psnr_sum = 0.0

        with torch.no_grad():
            for mosaic, pan, hrhsi in val_dataloader:
                mosaic = mosaic.to(device, non_blocking=True)
                pan = pan.to(device, non_blocking=True)
                hrhsi = hrhsi.to(device, non_blocking=True)

                prediction = fuse_net(mosaic, pan)
                psnr_sum += quality_index.calc_psnr(hrhsi, prediction).item()

        psnr_avg = psnr_sum / len(val_dataloader)
        spectral_response = pan_degradation.get_spectral_response().detach().flatten().cpu()

        checkpoint = {
            "epoch": epoch,
            "fuse_net": fuse_net.state_dict(),
            "pan_degradation": pan_degradation.state_dict(),
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
                "loss_mosaic": loss_mosaic_avg,
                "loss_pan": loss_pan_avg,
                "psnr": psnr_avg,
                "best_psnr": best_psnr,
                "best_epoch": best_epoch,
                "pan_response": spectral_response.tolist(),
            })

            with open(args.record_path, "w") as file:
                json.dump(record, file, indent=2)

        print(
            f"Epoch: {epoch} | loss: {loss_avg:.6f} | mosaic: {loss_mosaic_avg:.6f} | "
            f"pan: {loss_pan_avg:.6f} | psnr: {psnr_avg:.4f} | best: {best_psnr:.4f} "
            f"({best_epoch}) | time: {(time.time() - epoch_start_time) / 60:.2f} min"
        )
        print("PAN response:", numpy.array2string(spectral_response.numpy(), precision=4, suppress_small=True))

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
    experiment_dir = create_experiment_dir(args)
    args.cache_path = experiment_dir
    args.dir_model = os.path.join(experiment_dir, "model")
    args.record_path = os.path.join(experiment_dir, "record.json")

    os.makedirs(args.dir_model, exist_ok=True)

    if not args.resume and args.record and os.path.exists(args.record_path):
        os.remove(args.record_path)

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

    optimizer_params = [{"params": fuse_net.parameters(), "lr": args.lr}]

    if args.optimize_srf:
        optimizer_params.append({"params": pan_degradation.parameters(), "lr": args.lr_response})
    else:
        true_response = torch.tensor([1., 1, 2, 4, 8, 9, 10, 12, 16, 12, 10, 9, 7, 3, 2, 1], device=device)
        true_response = true_response / true_response.sum()
        with torch.no_grad():
            pan_degradation.response_logits.copy_(torch.log(true_response).view(1, -1, 1, 1))
        pan_degradation.response_logits.requires_grad_(False)

    optimizer = torch.optim.Adam(optimizer_params)

    args.start_epoch = 0
    args.best_epoch = 0
    args.best_psnr = float("-inf")

    if args.resume:
        checkpoint_path = args.resume if os.path.isfile(args.resume) else os.path.join(args.dir_model, args.resume)
        print("==> Load checkpoint:", checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        fuse_net.load_state_dict(checkpoint["fuse_net"])
        pan_degradation.load_state_dict(checkpoint["pan_degradation"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        args.start_epoch = checkpoint.get("epoch", 0)
        args.best_epoch = checkpoint.get("best_epoch", args.start_epoch)
        args.best_psnr = checkpoint.get("best_psnr", float("-inf"))
    else:
        print("==> Train from scratch")

    train(fuse_net, mosaic_degradation, pan_degradation, optimizer, train_dataloader, val_dataloader, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train mosaic and PAN fusion model")
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
    parser.add_argument("--lr_response", type=float, default=1e-5)
    parser.add_argument("--optimize_srf", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lambda_pan", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--data_path", type=str, default=os.path.expanduser("~/autodl-pub/Masico"))
    parser.add_argument("--record", action=argparse.BooleanOptionalAction, default=True)

    main(parser.parse_args())
