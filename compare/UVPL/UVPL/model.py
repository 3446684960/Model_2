import torch
import torch.nn as nn
import math
import numpy
import utils
import random

def transform(data, msfa_size=4, spatial_ratio=2):
    # rotation
    if numpy.random.uniform() < 0.5:
        rn = random.randint(1, 3)
        data = torch.rot90(data, rn, [2, 3])
 
    # flip
    if numpy.random.uniform() < 0.5:
        r = numpy.random.uniform()
        if r < 0.25:
            data = torch.flip(data, [2])
        elif r < 0.5:
            data = torch.flip(data, [3])
        elif r < 0.75:
            data = torch.flip(data, [2])
            data = torch.flip(data, [3])
        else:
            data = torch.flip(data, [3])
            data = torch.flip(data, [2])

    # shift
    if numpy.random.uniform() < 0.5:
        i = random.randint(0, msfa_size*spatial_ratio-1)
        j = random.randint(0, msfa_size*spatial_ratio-1)
        data = torch.roll(data, (i, j), (2, 3))

    return data

class Network(nn.Module):
    def __init__(self, args):
        super(Network, self).__init__()
        self.D = 3
        self.channels = 64
        self.msfa_size = [4, 4]

        self.conv1 = nn.Conv2d(args.num_bands+1, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.conv3 = nn.Conv2d(64, args.num_bands, 3, 1, 1)
        self.relu = nn.ReLU()
    
    def forward(self, mosaic, rgb):
        mosaic_reshape = torch.nn.functional.pixel_unshuffle(mosaic, downscale_factor=4)
        mosaic_interpolate = torch.nn.functional.interpolate(mosaic_reshape, scale_factor=8, mode="bilinear")
        x = torch.cat((mosaic_interpolate, rgb), 1)
        y = self.conv3(self.relu(self.conv2(self.relu(self.conv1(x))))) + rgb.mean(1, keepdims=True).repeat(1, mosaic_interpolate.shape[1], 1, 1)# + mosaic_interpolate
        return y

class CALayer(nn.Module):
    def __init__(self, nf, reduction=16):
        super(CALayer, self).__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.body = nn.Sequential(
            nn.Conv2d(nf, nf // reduction, 1, 1, 0),
            nn.LeakyReLU(0.2),
            nn.Conv2d(nf // reduction, nf, 1, 1, 0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.avg(x)
        y = self.body(y)
        return torch.mul(x, y)

class RB_Layer(nn.Module):
    def __init__(self, nf):
        super(RB_Layer, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(nf, nf, 3, 1, 1),
                                  nn.LeakyReLU(0.2, True),
                                  nn.Conv2d(nf, nf, 3, 1, 1),
                                  CALayer(nf))

    def forward(self, x):
        out = self.body(x) + x
        return out

class KernelNet(nn.Module):
    def __init__(self, args, in_nc=17, out_chn=16, nf=64, num_blocks=8, scale=4):
        super(KernelNet, self).__init__()

        self.head = nn.Conv2d(in_nc, nf, kernel_size=9, stride=4, padding=4, bias=False)

        self.body = nn.Sequential(*[RB_Layer(nf) for ii in range(num_blocks)])

        self.tail = nn.Sequential(nn.Conv2d(nf, out_chn, 3, 1, 1),
                                  nn.AdaptiveAvgPool2d((1, 1)))
        self.log_max = math.log(1e4)
        self.log_min = math.log(1e-4)

    def forward(self, x):
        x_head = self.head(x)
        x_body = self.body(x_head)
        out = self.tail(x_body)                     # N x 3 x 1 x 1
        lam12 = torch.exp(torch.clamp(out[:, :2,], min=self.log_min, max=self.log_max)) # N x 2 x 1 x 1
        # rho = torch.tanh(out[:, -1, ]).unsqueeze(1) # N x 1 x 1 x 1
        rho = out[:, -1, ].unsqueeze(1)             # N x 1 x 1 x 1
        Lam = torch.cat((lam12, rho), dim=1)        # N x 3 x 1 x 1
        Lam = torch.mean(Lam, dim=0, keepdims=True)
        return Lam

class SRFNet(nn.Module):
    def __init__(self, args, in_nc=17, out_chn=16, nf=64, num_blocks=8):
        super(SRFNet, self).__init__()

        self.out_chn = out_chn
        self.head = nn.Conv2d(in_nc, nf, kernel_size=9, stride=4, padding=4, bias=False)

        self.body = nn.Sequential(*[RB_Layer(nf) for ii in range(num_blocks)])

        self.tail = nn.Sequential(nn.Conv2d(nf, out_chn, 3, 1, 1),
                                  nn.AdaptiveAvgPool2d((1, 1)))
        
        self.log_max = math.log(1e4)
        self.log_min = math.log(1e-4)

    def forward(self, x):
        x_head = self.head(x)
        x_body = self.body(x_head)
        out = self.tail(x_body)                     # N x C x 1 x 1
        # R = torch.exp(torch.clamp(out, min=self.log_min, max=self.log_max))
        # R = torch.sigmoid(out)
        R = out.flatten(1)
        b, c = R.shape
        R = R.view(b, 1, c)
        R = torch.mean(R, dim=0)
        # R = torch.softmax(R, dim=-1)
        return R

def conv3x3(in_chn, out_chn, stride=1, bias=True):
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=3, stride=stride, padding=1, bias=bias)
    return layer

class DnCNN_y(nn.Module):
    def __init__(self, args, in_channels, out_channels, dep=3, num_filters=64, noise_avg=False):
        '''
        Reference:
        K. Zhang, W. Zuo, Y. Chen, D. Meng and L. Zhang, "Beyond a Gaussian Denoiser: Residual
        Learning of Deep CNN for Image Denoising," TIP, 2017.

        Args:
            in_channels (int): number of input channels
            out_channels (int): number of output channels
            dep (int): depth of the network, Default 8
            num_filters (int): number of filters in each layer, Default 64
        '''
        super(DnCNN_y, self).__init__()
        self.conv1 = conv3x3(in_channels, num_filters, bias=True)
        self.relu = nn.LeakyReLU(0.25, True)
        mid_layer = []
        for ii in range(1, dep-1):
            mid_layer.append(conv3x3(num_filters, num_filters, bias=True))
            mid_layer.append(nn.LeakyReLU(0.25, True))
        self.mid_layer = nn.Sequential(*mid_layer)
        self.conv_last = conv3x3(num_filters, out_channels, bias=True)
        if noise_avg:
            self.global_avg = nn.AdaptiveAvgPool2d((1,1))
        else:
            self.global_avg = nn.Identity()
        self.log_max = math.log(1e4)
        self.log_min = math.log(1e-4)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.mid_layer(x)
        x = self.conv_last(x)
        out = self.global_avg(x)
        out = torch.exp(torch.clamp(out, min=self.log_min, max=self.log_max))
        return out

class DnCNN_z(nn.Module):
    def __init__(self, args, in_channels, out_channels, dep=3, num_filters=64, noise_avg=False):
        '''
        Reference:
        K. Zhang, W. Zuo, Y. Chen, D. Meng and L. Zhang, "Beyond a Gaussian Denoiser: Residual
        Learning of Deep CNN for Image Denoising," TIP, 2017.

        Args:
            in_channels (int): number of input channels
            out_channels (int): number of output channels
            dep (int): depth of the network, Default 8
            num_filters (int): number of filters in each layer, Default 64
        '''
        super(DnCNN_z, self).__init__()
        self.conv1 = conv3x3(in_channels, num_filters, bias=True)
        self.relu = nn.LeakyReLU(0.25, True)
        mid_layer = []
        for ii in range(1, dep-1):
            mid_layer.append(conv3x3(num_filters, num_filters, stride=2, bias=True))
            mid_layer.append(nn.LeakyReLU(0.25, True))
        self.mid_layer = nn.Sequential(*mid_layer)
        self.conv_last = conv3x3(num_filters, out_channels, bias=True)
        if noise_avg:
            self.global_avg = nn.AdaptiveAvgPool2d((1,1))
        else:
            self.global_avg = nn.Identity()
        self.log_max = math.log(1e4)
        self.log_min = math.log(1e-4)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.mid_layer(x)
        x = self.conv_last(x)
        out = self.global_avg(x)
        out = torch.exp(torch.clamp(out, min=self.log_min, max=self.log_max))

        return out

def degrade_md(hrms, msfa_kernel, noise=None):
    x = torch.nn.functional.conv2d(hrms, msfa_kernel, bias=None, stride=msfa_kernel.shape[2], groups=hrms.shape[1])
    x = torch.nn.functional.pixel_shuffle(x, 4)
    if noise != None:
        x = x + noise
    return x

def degrade_srf(hrms, srf, noise=None):
    x = torch.einsum("bchw, nc ->bnhw", hrms, srf)
    if noise != None:
        x = x + noise
    return x