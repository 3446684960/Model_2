import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):

    def __init__(self, channels):
        super().__init__()

        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return x + self.body(x)


class PhaseAwareMSFAReconstruction(nn.Module):

    def __init__(self, num_bands=16, msfa_size=4, msfa_pattern=None):
        super().__init__()

        if msfa_pattern is None:
            msfa_pattern = torch.arange(num_bands).reshape(msfa_size, msfa_size)
        else:
            msfa_pattern = torch.as_tensor(msfa_pattern, dtype=torch.long)

        if msfa_pattern.shape != (msfa_size, msfa_size):
            raise ValueError(f"msfa_pattern must have shape [{msfa_size}, {msfa_size}].")

        if sorted(msfa_pattern.flatten().tolist()) != list(range(num_bands)):
            raise ValueError("msfa_pattern must contain every band index exactly once.")

        self.num_bands = num_bands
        self.msfa_size = msfa_size

        # pixel_unshuffle按照空间位置排列通道，这里将其重新排列成波段顺序。
        phase_to_band = msfa_pattern.flatten()
        band_to_phase = torch.argsort(phase_to_band)

        offsets = torch.zeros(num_bands, 2, dtype=torch.float32)
        for row in range(msfa_size):
            for col in range(msfa_size):
                band = int(msfa_pattern[row, col])
                offsets[band] = torch.tensor([row, col], dtype=torch.float32)

        self.register_buffer("band_to_phase", band_to_phase)
        self.register_buffer("offsets", offsets)

    def forward(self, mosaic, target_size):
        if mosaic.ndim != 4 or mosaic.shape[1] != 1:
            raise ValueError(f"mosaic must have shape [B, 1, H, W], but got {tuple(mosaic.shape)}.")

        if mosaic.shape[-2] % self.msfa_size != 0 or mosaic.shape[-1] % self.msfa_size != 0:
            raise ValueError("Mosaic height and width must be divisible by msfa_size.")

        batch_size, _, mosaic_height, mosaic_width = mosaic.shape

        # 无损拆出4×4 MSFA的16个空间相位。
        phase_cube = F.pixel_unshuffle(mosaic, downscale_factor=self.msfa_size)

        # 根据已知滤光片排列，将空间相位通道转换为波段通道。
        coarse_cube = phase_cube[:, self.band_to_phase]
        coarse_height, coarse_width = coarse_cube.shape[-2:]

        if coarse_height < 2 or coarse_width < 2:
            raise ValueError("The unpacked mosaic size is too small for phase-aware interpolation.")

        y = torch.arange(mosaic_height, device=mosaic.device, dtype=mosaic.dtype)
        x = torch.arange(mosaic_width, device=mosaic.device, dtype=mosaic.dtype)
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")

        row_offsets = self.offsets[:, 0].to(dtype=mosaic.dtype).view(self.num_bands, 1, 1)
        col_offsets = self.offsets[:, 1].to(dtype=mosaic.dtype).view(self.num_bands, 1, 1)

        coarse_y = (grid_y.unsqueeze(0) - row_offsets) / self.msfa_size
        coarse_x = (grid_x.unsqueeze(0) - col_offsets) / self.msfa_size

        normalized_y = 2.0 * coarse_y / (coarse_height - 1) - 1.0
        normalized_x = 2.0 * coarse_x / (coarse_width - 1) - 1.0
        sampling_grid = torch.stack([normalized_x, normalized_y], dim=-1)

        coarse_cube = coarse_cube.reshape(batch_size * self.num_bands, 1, coarse_height, coarse_width)
        sampling_grid = sampling_grid.unsqueeze(0).expand(batch_size, -1, -1, -1, -1)
        sampling_grid = sampling_grid.reshape(batch_size * self.num_bands, mosaic_height, mosaic_width, 2)

        aligned_cube = F.grid_sample(coarse_cube, sampling_grid, mode="bilinear", padding_mode="border", align_corners=True)
        aligned_cube = aligned_cube.reshape(batch_size, self.num_bands, mosaic_height, mosaic_width)

        # Mosaic分辨率到PAN分辨率，例如H/2×W/2到H×W。
        aligned_cube = F.interpolate(aligned_cube, size=target_size, mode="bilinear", align_corners=False)

        return aligned_cube

class SpectralMixingBlock(nn.Module):

    def __init__(self, num_bands=16, reduction=4):
        super().__init__()

        hidden_channels = num_bands * 2
        attention_channels = max(num_bands // reduction, 1)

        # 1x1 convolution mixes spectral bands without changing spatial positions.
        self.spectral_mixing = nn.Sequential(
            nn.Conv2d(num_bands, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, num_bands, kernel_size=1),
        )

        # Estimate a global weight for each spectral band.
        self.spectral_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_bands, attention_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(attention_channels, num_bands, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x + self.spectral_mixing(x)
        spectral_weights = self.spectral_attention(x)
        x = x * spectral_weights

        return x, spectral_weights


class MosaicSpectralBranch(nn.Module):

    def __init__(self, num_bands=16, msfa_size=4, msfa_pattern=None, feature_channels=64, num_blocks=3):
        super().__init__()

        if num_bands != msfa_size ** 2:
            raise ValueError(f"num_bands must equal msfa_size ** 2, but got {num_bands} and {msfa_size}.")

        self.num_bands = num_bands
        self.msfa_size = msfa_size

        self.spectral_mixing = SpectralMixingBlock(num_bands=num_bands)

        self.msfa_reconstruction = PhaseAwareMSFAReconstruction(
            num_bands=num_bands,
            msfa_size=msfa_size,
            msfa_pattern=msfa_pattern,
        )

        self.feature_head = nn.Sequential(
            nn.Conv2d(num_bands, feature_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )

        self.feature_body = nn.Sequential(
            *[ResidualBlock(feature_channels) for _ in range(num_blocks)]
        )

        self.base_head = nn.Conv2d(feature_channels, num_bands, kernel_size=3, padding=1)

    def forward(self, mosaic, target_size):
        if mosaic.ndim != 4 or mosaic.shape[1] != 1:
            raise ValueError(f"mosaic must have shape [B, 1, H, W], but got {tuple(mosaic.shape)}.")

        if mosaic.shape[-2] % self.msfa_size != 0 or mosaic.shape[-1] % self.msfa_size != 0:
            raise ValueError("Mosaic height and width must be divisible by msfa_size.")

        mosaic_cube = self.msfa_reconstruction(mosaic, target_size)

        spectral_cube, spectral_weights = self.spectral_mixing(mosaic_cube)
        spectral_feature = self.feature_head(spectral_cube)
        spectral_feature = self.feature_body(spectral_feature)
        base_logits = self.base_head(spectral_feature)

        return spectral_feature, base_logits, spectral_weights


class PanSpatialBranch(nn.Module):

    def __init__(self, feature_channels=64, branch_channels=32, num_blocks=2):
        super().__init__()

        self.branch_3x3 = nn.Sequential(
            nn.Conv2d(1, branch_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )

        self.branch_5x5 = nn.Sequential(
            nn.Conv2d(1, branch_channels, kernel_size=5, padding=2),
            nn.GELU(),
        )

        self.branch_7x7 = nn.Sequential(
            nn.Conv2d(1, branch_channels, kernel_size=7, padding=3),
            nn.GELU(),
        )

        self.feature_fusion = nn.Sequential(
            nn.Conv2d(branch_channels * 3, feature_channels, kernel_size=1),
            nn.GELU(),
            *[ResidualBlock(feature_channels) for _ in range(num_blocks)],
        )

    def forward(self, pan):
        if pan.ndim != 4 or pan.shape[1] != 1:
            raise ValueError(f"pan must have shape [B, 1, H, W], but got {tuple(pan.shape)}.")

        feature_3x3 = self.branch_3x3(pan)
        feature_5x5 = self.branch_5x5(pan)
        feature_7x7 = self.branch_7x7(pan)

        multi_scale_feature = torch.cat([feature_3x3, feature_5x5, feature_7x7], dim=1)
        spatial_feature = self.feature_fusion(multi_scale_feature)

        return spatial_feature

class FusionNet(nn.Module):

    def __init__(self, num_bands=16, msfa_size=4, spatial_ratio=2, feature_channels=64):
        super().__init__()

        self.num_bands = num_bands
        self.spatial_ratio = spatial_ratio

        self.mosaic_branch = MosaicSpectralBranch(
            num_bands=num_bands,
            msfa_size=msfa_size,
            feature_channels=feature_channels,
        )

        self.pan_branch = PanSpatialBranch(feature_channels=feature_channels)

        self.fusion_body = nn.Sequential(
            nn.Conv2d(feature_channels * 2, feature_channels, kernel_size=3, padding=1),
            nn.GELU(),
            ResidualBlock(feature_channels),
            ResidualBlock(feature_channels),
        )

        # Estimate how much PAN detail should be injected into each spectral band.
        self.band_gate = nn.Sequential(
            nn.Conv2d(feature_channels, num_bands, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

        self.detail_head = nn.Conv2d(feature_channels, num_bands, kernel_size=3, padding=1)

    def forward(self, mosaic, pan, return_aux=False):
        if mosaic.ndim != 4 or pan.ndim != 4:
            raise ValueError("mosaic and pan must be 4D tensors.")

        if mosaic.shape[0] != pan.shape[0]:
            raise ValueError("mosaic and pan must have the same batch size.")

        if pan.shape[-2] != mosaic.shape[-2] * self.spatial_ratio or pan.shape[-1] != mosaic.shape[-1] * self.spatial_ratio:
            raise ValueError("PAN spatial size must be spatial_ratio times the mosaic spatial size.")

        target_size = pan.shape[-2:]

        spectral_feature, base_logits, spectral_weights = self.mosaic_branch(mosaic, target_size)
        spatial_feature = self.pan_branch(pan)

        fusion_feature = torch.cat([spectral_feature, spatial_feature], dim=1)
        fusion_feature = self.fusion_body(fusion_feature)

        band_gate = self.band_gate(fusion_feature)
        pan_detail = self.detail_head(spatial_feature)

        # Base spectral reconstruction + selectively injected PAN detail.
        output_logits = base_logits + band_gate * pan_detail
        hrhsi = torch.sigmoid(output_logits)

        if return_aux:
            auxiliary = {
                "spectral_weights": spectral_weights,
                "band_gate": band_gate,
            }
            return hrhsi, auxiliary

        return hrhsi