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
        low_hsi = F.avg_pool2d(hrhsi, kernel_size=self.spatial_ratio, stride=self.spatial_ratio)
        return self.sample_msfa(low_hsi)

    def sample_msfa(self, hsi):
        if hsi.ndim != 4 or hsi.shape[1] != self.num_bands:
            raise ValueError(f"hsi must have shape [B, {self.num_bands}, H, W].")

        low_height, low_width = hsi.shape[-2:]
        if (low_height % self.msfa_size != 0 or low_width % self.msfa_size != 0):
            raise ValueError("The low-resolution HSI size must be divisible " "by msfa_size.")

        # Repeat the 4x4 MSFA pattern over the whole image.
        repeat_h = low_height // self.msfa_size
        repeat_w = low_width // self.msfa_size

        mask = self.msfa_mask.repeat(1, 1, repeat_h, repeat_w)

        # Each spatial position keeps only one spectral band.
        return torch.sum(hsi * mask, dim=1, keepdim=True)


def create_wald_inputs(mosaic, pan, msfa_size=4, spatial_ratio=2, generator=None):
    batch, _, height, width = mosaic.shape
    alignment = msfa_size * spatial_ratio
    if height % alignment or width % alignment or min(height, width) < 2 * alignment:
        raise ValueError("Wald Mosaic dimensions must contain at least two coarse MSFA tiles and align with the scale.")
    if pan.shape != (batch, 1, height * spatial_ratio, width * spatial_ratio):
        raise ValueError("PAN and Mosaic shapes do not match the Wald spatial ratio.")

    phases = F.pixel_unshuffle(mosaic, msfa_size)
    num_phases, phase_height, phase_width = phases.shape[1:]
    rows = torch.arange(phase_height, device=mosaic.device)
    cols = torch.arange(phase_width, device=mosaic.device)
    parity = (rows[:, None] % 2) * 2 + cols[None, :] % 2
    held_parity = torch.randint(4, (batch, num_phases, 1, 1), device=mosaic.device, generator=generator)
    held_out = parity == held_parity

    # 同相位低通，目标读数不参与滤波；同时滤波权重以归一化有效样本。
    values = phases.masked_fill(held_out, 0)
    observed = (~held_out).to(mosaic.dtype)
    filtered = F.avg_pool2d(torch.stack((values, observed), dim=2).flatten(0, 1), 3, 1, 1)

    # 粗网格像素 j 的中心落在原网格 s*j+(s-1)/2，再换算到对应相位的采样坐标。
    row_phase = torch.arange(msfa_size, device=mosaic.device, dtype=mosaic.dtype).repeat_interleave(msfa_size)
    col_phase = torch.arange(msfa_size, device=mosaic.device, dtype=mosaic.dtype).repeat(msfa_size)
    row_offset = (spatial_ratio - 1) * (row_phase + 0.5) / msfa_size
    col_offset = (spatial_ratio - 1) * (col_phase + 0.5) / msfa_size
    y = rows[:phase_height // spatial_ratio] * spatial_ratio + row_offset[:, None]
    x = cols[:phase_width // spatial_ratio] * spatial_ratio + col_offset[:, None]
    y = (2 * y / (phase_height - 1) - 1).unsqueeze(-1)
    x = (2 * x / (phase_width - 1) - 1).unsqueeze(-2)
    grid_x, grid_y = torch.broadcast_tensors(x, y)
    grid = torch.stack((grid_x, grid_y), dim=-1).repeat(batch, 1, 1, 1)
    samples = F.grid_sample(filtered, grid, padding_mode="border", align_corners=True)
    coarse_phases = (samples[:, 0] / samples[:, 1].clamp_min(1e-6)).unflatten(0, (batch, num_phases))

    coarse_mosaic = F.pixel_shuffle(coarse_phases, msfa_size)
    coarse_pan = F.avg_pool2d(pan, spatial_ratio, spatial_ratio)
    holdout_mask = F.pixel_shuffle(held_out.to(mosaic.dtype), msfa_size).bool()
    return coarse_mosaic, coarse_pan, holdout_mask



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
