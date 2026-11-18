import torch
import torch.nn as nn
import torch.nn.functional as F


class MosaicDegradation(nn.Module):

    def __init__(self, num_bands=16, msfa_size=4, spatial_ratio=2):
        super().__init__()

        if num_bands != msfa_size ** 2:
            raise ValueError("num_bands must equal msfa_size ** 2, "f"but got {num_bands} and {msfa_size}.")

        self.num_bands = num_bands
        self.msfa_size = msfa_size
        self.spatial_ratio = spatial_ratio

        # Shape: [1, num_bands, msfa_size, msfa_size]
        msfa_mask = torch.zeros(1, num_bands, msfa_size, msfa_size, dtype=torch.float32,)

        band = 0
        for row in range(msfa_size):
            for col in range(msfa_size):
                msfa_mask[0, band, row, col] = 1.0
                band += 1

        # mask is fixed and is moved automatically with the model.
        self.register_buffer("msfa_mask", msfa_mask)

    def forward(self, hrhsi):

        if hrhsi.ndim != 4:
            raise ValueError("hrhsi must be a 4D tensor [B, C, H, W].")

        _, channels, height, width = hrhsi.shape

        if channels != self.num_bands:
            raise ValueError(f"Expected {self.num_bands} bands, "f"but got {channels}.")

        if (height % self.spatial_ratio != 0 or width % self.spatial_ratio != 0):
            raise ValueError("HRHSI height and width must be divisible " "by spatial_ratio.")

        # Spatial degradation:
        # [B, 16, H, W] -> [B, 16, H/2, W/2]   #先不建模PSF
        low_hsi = F.avg_pool2d(hrhsi, kernel_size=self.spatial_ratio, stride=self.spatial_ratio,)

        low_height = low_hsi.shape[-2]
        low_width = low_hsi.shape[-1]

        if (low_height % self.msfa_size != 0 or low_width % self.msfa_size != 0):
            raise ValueError("The low-resolution HSI size must be divisible " "by msfa_size.")

        # Repeat the 4x4 MSFA pattern over the whole image.
        repeat_h = low_height // self.msfa_size
        repeat_w = low_width // self.msfa_size

        mask = self.msfa_mask.repeat(1, 1, repeat_h, repeat_w)

        # Each spatial position keeps only one spectral band.
        mosaic = torch.sum(low_hsi * mask, dim=1, keepdim=True,)

        return mosaic



class PanDegradation(nn.Module):

    def __init__(self, num_bands=16):
        super().__init__()

        self.num_bands = num_bands

        self.response_logits = nn.Parameter(
            torch.zeros(1, num_bands, 1, 1, dtype=torch.float32)
        )

    def get_spectral_response(self):
        """Return the non-negative, normalized PAN spectral response."""
        return torch.softmax(self.response_logits, dim=1)

    def forward(self, hrhsi):
        if hrhsi.ndim != 4:
            raise ValueError("hrhsi must be a 4D tensor [B, C, H, W], "f"but got {tuple(hrhsi.shape)}.")

        if hrhsi.shape[1] != self.num_bands:
            raise ValueError(f"Expected {self.num_bands} spectral bands, "f"but got {hrhsi.shape[1]}.")

        spectral_response = self.get_spectral_response()
        pan = F.conv2d(hrhsi, spectral_response)

        return pan
