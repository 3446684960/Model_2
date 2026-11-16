import time
import os
import torch
from torch.utils.data import DataLoader
from GetDataSet import MakeRealWorldDataset
import torch.nn as nn
from model import Network, SRFNet, KernelNet, DnCNN_y, DnCNN_z, degrade_md, degrade_srf, transform
import copy
import argparse
import json
import cv2
import numpy
import utils
import random, tqdm
import quality_index
import math

def train(fuse_net: nn.Module, RNet: nn.Module, SNet_z: nn.Module, SNet_y: nn.Module, 
            optimizer, train_dataloader, val_dataloader, args):
    print('===>Begin Training!')
    start_epoch = 0
    if args.resume != "":
        start_epoch = int(args.resume) if "best" not in args.resume else int(args.resume.split("_")[-1])

    t = time.time()
    device = next(fuse_net.parameters()).device
    MSFA = numpy.array([[0, 1, 2, 3],
                        [4, 5, 6, 7],
                        [8, 9, 10, 11],
                        [12, 13, 14, 15]])
    msfa_kernel = torch.zeros(MSFA.shape[0] * MSFA.shape[1], 1, MSFA.shape[0]*2, MSFA.shape[1]*2).to(device)
    for i in range(MSFA.shape[0]):
        for j in range(MSFA.shape[1]):
            msfa_kernel[int(MSFA[i, j]), 0, i*2, j*2] = 0.25
            msfa_kernel[int(MSFA[i, j]), 0, i*2+1, j*2] = 0.25
            msfa_kernel[int(MSFA[i, j]), 0, i*2, j*2+1] = 0.25
            msfa_kernel[int(MSFA[i, j]), 0, i*2+1, j*2+1] = 0.25
    best_qnr = 0
    best_epoch = 0
    numpy.set_printoptions(precision=3, suppress=True)

    m_z = torch.tensor(0., dtype=torch.float32).to(device)
    m_y = torch.tensor(0., dtype=torch.float32).to(device)
    alpha_z = torch.tensor(100, dtype=torch.float32).to(device)
    alpha_y = torch.tensor(100, dtype=torch.float32).to(device)
    sigma_r = torch.tensor(0.001, dtype=torch.float32).to(device)
    sigma_x = torch.tensor(0.001, dtype=torch.float32).to(device)

    mu_srf = torch.ones(1, args.num_bands).to(device)
    mu_srf = mu_srf / args.num_bands

    for epoch in range(start_epoch + 1, args.epochs + 1):
        # train
        fuse_net.train()
        RNet.train()
        SNet_z.train()
        SNet_y.train()
        start_time = time.time()

        loss_per_epoch = 0
        for cnt, data in enumerate(tqdm.tqdm(train_dataloader)):
            if cnt == 1000:
                break
            z, y = data[0].to(device), data[1].to(device)
            up_z = torch.nn.functional.interpolate(torch.nn.functional.pixel_unshuffle(z, msfa_kernel.shape[-1]//2), scale_factor=msfa_kernel.shape[-1], mode="bicubic")
            cat = torch.cat((up_z, y), 1)
            x_u_first = fuse_net(z, y)
            x_first = x_u_first + sigma_x * torch.randn_like(x_u_first, dtype=torch.float32).to(device)

            # Nz, Ny are assumed to be Gaussian distribution with knowing u=0
            # and their variance are assumed to be inverse Gamma distribution with knowing alpha, i. e., alpha_1, alpha_2
            beta_z_first = SNet_z(cat)
            beta_y_first = SNet_y(cat)

            # srf is assumed to be Gaussian distribution with knowing u
            mu_srf_first = RNet(cat)

            # Nz_var is sampled from the 1 / Gamma distribution
            Nz_var_est = 1 / torch.distributions.Gamma(alpha_z, beta_z_first).rsample()
            # using the reparameterization trick to estimate Nz
            Nz_est = m_z + Nz_var_est.sqrt() * torch.randn_like(Nz_var_est, dtype=torch.float32).to(device)

            # Ny_var is sampled from the 1 / Gamma distribution
            Ny_var_est = 1 / torch.distributions.Gamma(alpha_y, beta_y_first).rsample()
            # using the reparameterization trick to estimate Ny
            Ny_est = m_y + Ny_var_est.sqrt() * torch.randn_like(Ny_var_est, dtype=torch.float32).to(device)

            # using the reparameterization trick to estimate srf
            srf_est = mu_srf_first + sigma_r * torch.randn_like(mu_srf, dtype=torch.float32).to(device)

            # degrade
            z_degrade = degrade_md(x_first, msfa_kernel, Nz_est)
            y_degrade = degrade_srf(x_first, srf_est, Ny_est)

            # loss function
            loss_exp = 0.5*((z_degrade - z)**2).mean() + 0.5*((y_degrade - y)**2).mean()

            # transform
            cat_tmp = transform(torch.cat((x_first, cat, x_u_first), 1), msfa_size=4, spatial_ratio=2).detach()
            c1 = x_first.shape[1]
            c2 = cat.shape[1]
            c3 = x_u_first.shape[1]
            x_first, cat, x_u_first = cat_tmp[:, :c1], cat_tmp[:, c1:c1+c2], cat_tmp[:, c1+c2:c1+c2+c3]

            # estimate Nz and Ny once again
            beta_z_second = SNet_z(cat)
            Nz_var_est = 1 / torch.distributions.Gamma(alpha_z, beta_z_second).rsample()
            Nz_est = m_z + Nz_var_est.sqrt() * torch.randn_like(Nz_var_est, dtype=torch.float32).to(device)
            beta_y_second = SNet_y(cat)
            Ny_var_est = 1 / torch.distributions.Gamma(alpha_y, beta_y_second).rsample()
            Ny_est = m_y + Ny_var_est.sqrt() * torch.randn_like(Ny_var_est, dtype=torch.float32).to(device)

            # predict srf_var_beta again
            mu_srf_second = RNet(cat)

            # degrade again
            z_degrade = degrade_md(x_first, msfa_kernel, Nz_est)
            y_degrade = degrade_srf(x_first, srf_est, Ny_est)
            # inference again
            x_u_second = fuse_net(z_degrade, y_degrade)
            x_second = x_u_second + sigma_x * torch.randn_like(x_u_second, dtype=torch.float32).to(device)
            # loss function EI
            loss_ei = 0.5*((x_first - x_second)**2).mean()

            # compute KL divergence
            loss_Nz = alpha_z*((beta_z_second / beta_z_first).log() + (beta_z_first / beta_z_second) - 1).mean() # inverse Gamma
            loss_Ny = alpha_y*((beta_y_second / beta_y_first).log() + (beta_y_first / beta_y_second) - 1).mean() # inverse Gamma
            loss_R = 0.5*((mu_srf_second - mu_srf_first)**2 / sigma_r**2).mean() # Gaussian

            loss = loss_exp + loss_ei + loss_R + loss_Ny + loss_Nz
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_per_epoch += loss.detach().item()

        loss_per_epoch /= cnt + 1
        qnr_avg, d_lambda_avg, d_s_avg = 0., 0., 0.
        # val
        fuse_net.eval()
        with torch.no_grad():
            for cnt, data in enumerate(val_dataloader):
                z, y = data[0].to(device), data[1].to(device)
                up_z = torch.nn.functional.interpolate(torch.nn.functional.pixel_unshuffle(z, msfa_kernel.shape[-1]//2), scale_factor=msfa_kernel.shape[-1], mode="bicubic")
                cat = torch.cat((up_z, y), 1)
                x_u_first = fuse_net(z, y)
                x_first = x_u_first + sigma_x * torch.randn_like(x_u_first, dtype=torch.float32).to(device)

                beta_z_first = SNet_z(cat)
                beta_y_first = SNet_y(cat)

                mu_srf_first = RNet(cat)

                Nz_var_est = 1 / torch.distributions.Gamma(alpha_z, beta_z_first).rsample()
                Nz_est = m_z + Nz_var_est.sqrt() * torch.randn_like(Nz_var_est, dtype=torch.float32).to(device)

                Ny_var_est = 1 / torch.distributions.Gamma(alpha_y, beta_y_first).rsample()
                Ny_est = m_y + Ny_var_est.sqrt() * torch.randn_like(Ny_var_est, dtype=torch.float32).to(device)

                srf_est = mu_srf_first + sigma_r * torch.randn_like(mu_srf, dtype=torch.float32).to(device)

                z_degrade = degrade_md(x_first, msfa_kernel, Nz_est)
                y_degrade = degrade_srf(x_first, srf_est, Ny_est)

                fused = torch.clamp(torch.round(x_first*2**12), 0, 2**12).short()                   
                
                qnr, d_lambda, d_s = quality_index.calc_qnr_mosaic(fused.double()/2**12, z.double(), y.double(), msfa_kernel, patch_size=128, scale_factor=args.spatial_ratio, p=0.3, q=1, alpha=0.8, beta=1.2)
                
                qnr_avg += qnr.item()
                d_lambda_avg += d_lambda.item()
                d_s_avg += d_s.item()

                if args.visual == True and epoch % args.visual_freq == 0 and epoch != 0:
                    rgb_bands = [0, 1, 13]
                    if not os.path.exists(os.path.join(args.dir_record, str(cnt))):
                        os.mkdir(os.path.join(args.dir_record, str(cnt)))

                    if not os.path.exists(os.path.join(args.dir_record, str(cnt), "z.png")):
                        z_numpy = utils.pixel_shuffle_inv(z.cpu().numpy(), 4)[0].transpose(1, 2, 0)[:, :, rgb_bands]
                        cv2.imwrite(os.path.join(args.dir_record, str(cnt), "z.png"), numpy.clip(numpy.round(z_numpy * 255), 0, 255))
                        y_numpy = y.cpu().numpy()[0].transpose(1, 2, 0)
                        cv2.imwrite(os.path.join(args.dir_record, str(cnt), "y.png"), numpy.clip(numpy.round(y_numpy * 255), 0, 255))    
                    z_degrade_numpy = utils.pixel_shuffle_inv(z_degrade.cpu().numpy(), 4)[0].transpose(1, 2, 0)[:, :, rgb_bands]
                    cv2.imwrite(os.path.join(args.dir_record, str(cnt), f"z_degrade_{epoch}.png"), numpy.clip(numpy.round(z_degrade_numpy * 255), 0, 255))
                    y_degrade_numpy = y_degrade.cpu().numpy()[0].transpose(1, 2, 0)
                    cv2.imwrite(os.path.join(args.dir_record, str(cnt), f"y_degrade_{epoch}.png"), numpy.clip(numpy.round(y_degrade_numpy * 255), 0, 255))
                    fused_numpy = fused[0].permute(1, 2, 0).cpu().numpy()[:, :, rgb_bands]
                    cv2.imwrite(os.path.join(args.dir_record, str(cnt), f"fused_{epoch}.png"), numpy.clip(numpy.round(fused_numpy / 2**12 * 255), 0, 255))

        qnr_avg /= cnt+1
        d_lambda_avg /= cnt+1
        d_s_avg /= cnt+1

        # save model with highest QNR
        if qnr_avg > best_qnr:
            best_qnr = qnr_avg
            if best_epoch != 0:
                os.remove(os.path.join(args.dir_model, "best_{}.pth".format(best_epoch)))
            best_epoch = epoch
            torch.save({"fuse_net": fuse_net.state_dict(),
                         "RNet": RNet.state_dict(), 
                         "SNet_y": SNet_y.state_dict(),
                         "SNet_z": SNet_z.state_dict()}, os.path.join(args.dir_model, "best_{}.pth".format(epoch)))
        
        if args.record is not False:
            record = []
            if os.path.exists(args.record):
                with open(args.record, "r") as f:
                    record = json.load(f)
            record.append({"epoch": epoch,
                           "loss": loss_per_epoch,
                           "qnr": qnr_avg,
                           "best_qnr": best_qnr,
                           "best_epoch": best_epoch,
                           "learning rate": optimizer.param_groups[0]["lr"],
                           })
            with open(args.record, "w") as f:
                record = json.dump(record, f, indent=2)
        
        # save model at some frequency
        if epoch % args.save_freq == 0:
            torch.save({"fuse_net": fuse_net.state_dict(),
                         "RNet": RNet.state_dict(), 
                         "SNet_y": SNet_y.state_dict(),
                         "SNet_z": SNet_z.state_dict()}, os.path.join(args.dir_model, f"{epoch}.pth"))

        # log
        print("Epoch: ", epoch,
            "loss: %.4f"%loss_per_epoch,
            "time: %.2f"%((time.time() - start_time) / 60), "min",
            "loss: %.4f"%loss_per_epoch,
            "qnr: %.4f"%qnr_avg,
            "best_qnr: %.4f"%best_qnr,
            "best_epoch: ", best_epoch,
            "learning rate: ", optimizer.param_groups[0]["lr"], "\n",
            )
        loss_per_epoch = 0 

    print(f"Total time: {(time.time() - t) / 60} min")
    print("Best epoch: {}, Best QNR: {}".format(best_epoch, best_qnr))

def main(args):
    dir_idx = os.path.join("./", str(args.idx))
    if not os.path.exists(dir_idx):
        os.mkdir(dir_idx)
    args.cache_path = dir_idx

    dir_model = os.path.join(dir_idx, "model")
    if not os.path.exists(dir_model):
        os.mkdir(dir_model)
    args.dir_model = dir_model

    if args.record is True:
        dir_record = os.path.join(dir_idx, "record")
        if not os.path.exists(dir_record):
            os.makedirs(dir_record)
        args.dir_record = dir_record
        args.record = os.path.join(dir_record, "record.json")
        if args.resume == "" and os.path.exists(args.record):
            os.remove(args.record)

    total_iterations = args.epochs * args.iters_per_epoch
    print('total_iterations:{}'.format(total_iterations))

    train_set = MakeRealWorldDataset(args, "train")
    train_dataloader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=True)
    val_set = MakeRealWorldDataset(args, "test")
    val_dataloader = DataLoader(dataset=val_set, batch_size=1, shuffle=False, num_workers=args.num_workers,
                                pin_memory=True)

    fuse_net = Network(args)
    RNet = SRFNet(args, 17, 16)
    SNet_z = DnCNN_z(args, 17, 1)
    SNet_y = DnCNN_y(args, 17, 1)

    if args.resume != "":
        backup_pth = os.path.join(dir_model, args.resume + ".pth")
        print("==> Load checkpoint: {}".format(backup_pth))
        fuse_net.load_state_dict(torch.load(backup_pth)["fuse_net"], strict=False)
        RNet.load_state_dict(torch.load(backup_pth)["RNet"], strict=False)
        SNet_z.load_state_dict(torch.load(backup_pth)["SNet_z"], strict=False)
        SNet_y.load_state_dict(torch.load(backup_pth)["SNet_y"], strict=False)
    else:
        print('==> Train from scratch')
    
    fuse_net = fuse_net.to(f"cuda:{args.device}")
    RNet = RNet.to(f"cuda:{args.device}")
    SNet_z = SNet_z.to(f"cuda:{args.device}")
    SNet_y = SNet_y.to(f"cuda:{args.device}")

    optimizer = torch.optim.Adam([
                                {"params": fuse_net.parameters(), "lr": args.lr},
                                {"params": RNet.parameters(), "lr": args.lr},
                                {"params": SNet_z.parameters(), "lr": args.lr},
                                {"params": SNet_y.parameters(), "lr": args.lr}
                            ])

    train(fuse_net, RNet, SNet_z, SNet_y, optimizer, train_dataloader, val_dataloader, args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('--idx', type=int, default=1, help='Index to identify models.')
    parser.add_argument('--dataset', type=str, default="Ours", help='Dataset to be loaded.')
    parser.add_argument('--train_size', type=int, default=128, help='Size of the training image in a batch.')
    parser.add_argument('--msfa_size', type=int, default=4, help='Size of MSFA')
    parser.add_argument('--spatial_ratio', type=int, default=2, help='Ratio of spatial resolutions between MS and PAN')
    parser.add_argument('--num_bands', type=int, default=16, help='Number of bands of a MS image.')
    parser.add_argument('--stride', type=int, default=64, help='Stride when crop an original image into patches.')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training dataset.')
    parser.add_argument('--epochs', type=int, default=1000, help='Total epochs to train the model.')
    parser.add_argument('--iters_per_epoch', type=int, default=100, help='Iteration steps per epoch.')
    parser.add_argument('--save_freq', type=int, default=50, help='Save the checkpoints of the model every [save_freq] epochs.')
    parser.add_argument('--optimizer', type=str, default="adam", help='Optimizer.')
    parser.add_argument('--lr_decay', action="store_true", help='Determine if to decay the learning rate.')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate to train the model.')
    parser.add_argument('--device', type=str, default='0', help='Device to train the model.')
    parser.add_argument('--num_workers', type=int, default=1, help='Num_workers to train the model.')
    parser.add_argument('--resume', type=str, default='', help='Index of the model to be resumed, eg. 1000.')
    parser.add_argument('--resume_demosaic', type=str, default='', help='Index of the demosaicing model to be resumed, eg. 1000.')
    parser.add_argument('--data_path', type=str, default="../DataSet/", help='Path of the dataset.')
    parser.add_argument('--record', type=bool, default=True, help='Whether to record the PSNR of each epoch.')
    parser.add_argument('--visual', action='store_true', default=False, help='Whether to visualize the validation data.')
    parser.add_argument('--visual_freq', type=int, default=50, help='Frequency to visualize the validation data.')

    args = parser.parse_args()
    main(args)