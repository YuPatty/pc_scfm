import math

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .factory import register_model
except ImportError:
    from factory import register_model


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        device = t.device
        scale = math.log(10000) / max(half - 1, 1)
        freqs = torch.exp(torch.arange(half, device=device, dtype=torch.float32) * -scale)
        emb = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.dim % 2:
            emb = F.pad(emb, (0, 1))
        return emb


class ConvBlock1d(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim, groups=8, dropout=0.0):
        super().__init__()
        group_count = min(groups, out_channels)
        while out_channels % group_count != 0:
            group_count -= 1
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(group_count, out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(group_count, out_channels)
        self.time_proj = nn.Linear(time_dim, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.skip = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, time_emb):
        residual = self.skip(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x = F.silu(x + self.time_proj(time_emb).unsqueeze(-1))
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.norm2(x)
        return F.silu(x + residual)


class DeepAggregationPyramidPooling1d(nn.Module):
    def __init__(self, channels, pool_scales=(2, 4, 8)):
        super().__init__()
        self.pool_scales = tuple(pool_scales)
        self.projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.AdaptiveAvgPool1d(scale),
                    nn.Conv1d(channels, channels, 1),
                    nn.SiLU(),
                )
                for scale in self.pool_scales
            ]
        )
        self.fuse = nn.Sequential(
            nn.Conv1d(channels * (len(self.pool_scales) + 1), channels, 1),
            nn.GroupNorm(1, channels),
            nn.SiLU(),
            nn.Conv1d(channels, channels, 3, padding=1),
        )

    def forward(self, x):
        length = x.shape[-1]
        features = [x]
        for projection in self.projections:
            pooled = projection(x)
            features.append(F.interpolate(pooled, size=length, mode="linear", align_corners=False))
        return self.fuse(torch.cat(features, dim=1)) + x


class DownBlock1d(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim, dropout=0.0):
        super().__init__()
        self.block = ConvBlock1d(in_channels, out_channels, time_dim, dropout=dropout)
        self.pool = nn.Conv1d(out_channels, out_channels, 4, stride=2, padding=1)

    def forward(self, x, time_emb):
        skip = self.block(x, time_emb)
        return self.pool(skip), skip


class UpBlock1d(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, time_dim, dropout=0.0):
        super().__init__()
        self.up = nn.ConvTranspose1d(in_channels, out_channels, 4, stride=2, padding=1)
        self.block = ConvBlock1d(out_channels + skip_channels, out_channels, time_dim, dropout=dropout)

    def forward(self, x, skip, time_emb):
        x = self.up(x)
        if x.shape[-1] != skip.shape[-1]:
            x = F.interpolate(x, size=skip.shape[-1], mode="linear", align_corners=False)
        return self.block(torch.cat([x, skip], dim=1), time_emb)


class EDDMUNet1d(nn.Module):
    def __init__(
        self,
        in_channels=2,
        base_channels=64,
        channel_mults=(1, 2, 4),
        time_dim=256,
        dropout=0.0,
        pool_scales=(2, 4, 8),
    ):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        channels = [base_channels * mult for mult in channel_mults]
        self.input = nn.Conv1d(in_channels, channels[0], 3, padding=1)
        self.downs = nn.ModuleList()
        in_ch = channels[0]
        for out_ch in channels:
            self.downs.append(DownBlock1d(in_ch, out_ch, time_dim, dropout=dropout))
            in_ch = out_ch

        self.mid = nn.ModuleList(
            [
                ConvBlock1d(channels[-1], channels[-1], time_dim, dropout=dropout),
                DeepAggregationPyramidPooling1d(channels[-1], pool_scales=pool_scales),
                ConvBlock1d(channels[-1], channels[-1], time_dim, dropout=dropout),
            ]
        )

        self.ups = nn.ModuleList()
        current = channels[-1]
        for skip_ch in reversed(channels):
            out_ch = skip_ch
            self.ups.append(UpBlock1d(current, skip_ch, out_ch, time_dim, dropout=dropout))
            current = out_ch

        self.output = nn.Sequential(
            nn.GroupNorm(1, channels[0]),
            nn.SiLU(),
            nn.Conv1d(channels[0], 2, 3, padding=1),
        )

    def forward(self, x, t):
        time_emb = self.time_mlp(t)
        x = self.input(x)
        skips = []
        for down in self.downs:
            x, skip = down(x, time_emb)
            skips.append(skip)
        for layer in self.mid:
            if isinstance(layer, ConvBlock1d):
                x = layer(x, time_emb)
            else:
                x = layer(x)
        for up, skip in zip(self.ups, reversed(skips)):
            x = up(x, skip, time_emb)
        return self.output(x)


@register_model("eddm")
class EDDMDenoiser(nn.Module):
    def __init__(
        self,
        timesteps=50,
        inference_steps=10,
        base_channels=64,
        channel_mults=(1, 2, 4),
        time_dim=256,
        dropout=0.0,
        gaussian_scale=0.2,
        ecg_noise_weight=1.0,
        gaussian_noise_weight=0.1,
        clean_weight=1.0,
        pcc_weight=0.1,
        pool_scales=(2, 4, 8),
        **kwargs,
    ):
        super().__init__()
        self.timesteps = int(timesteps)
        self.inference_steps = int(inference_steps)
        self.gaussian_scale = float(gaussian_scale)
        self.ecg_noise_weight = float(ecg_noise_weight)
        self.gaussian_noise_weight = float(gaussian_noise_weight)
        self.clean_weight = float(clean_weight)
        self.pcc_weight = float(pcc_weight)
        self.net = EDDMUNet1d(
            in_channels=2,
            base_channels=int(base_channels),
            channel_mults=tuple(channel_mults),
            time_dim=int(time_dim),
            dropout=float(dropout),
            pool_scales=tuple(pool_scales),
        )

        betas = torch.linspace(1.0e-4, 0.02, self.timesteps)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        gamma = torch.linspace(0.0, 1.0, self.timesteps)
        sigma = torch.sqrt(1.0 - alpha_bars) * self.gaussian_scale
        self.register_buffer("gamma", gamma)
        self.register_buffer("sigma", sigma)

    def _coefficients(self, t, length):
        gamma = self.gamma[t].view(-1, 1, 1)
        sigma = self.sigma[t].view(-1, 1, 1)
        return gamma, sigma

    def _predict(self, xt, noisy, t):
        pred = self.net(torch.cat([xt, noisy], dim=1), t)
        return pred[:, 0:1], pred[:, 1:2]

    def _clean_from_prediction(self, xt, t, ecg_noise, gaussian_noise):
        gamma, sigma = self._coefficients(t, xt.shape[-1])
        return xt - gamma * ecg_noise - sigma * gaussian_noise

    def forward(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(1)
        current = x
        steps = torch.linspace(
            self.timesteps - 1,
            0,
            steps=max(self.inference_steps, 1),
            device=x.device,
        ).round().long().unique(sorted=True).flip(0)
        for t_value in steps:
            t = torch.full((x.shape[0],), int(t_value.item()), device=x.device, dtype=torch.long)
            ecg_noise, gaussian_noise = self._predict(current, x, t)
            clean = self._clean_from_prediction(current, t, ecg_noise, gaussian_noise)
            prev_t = max(int(t_value.item()) - 1, 0)
            prev = torch.full((x.shape[0],), prev_t, device=x.device, dtype=torch.long)
            prev_gamma, _ = self._coefficients(prev, x.shape[-1])
            current = clean + prev_gamma * ecg_noise
        return current

    @torch.no_grad()
    def denoising(self, x):
        return self.forward(x)

    def compute_loss(self, batch, device, **kwargs):
        noisy, clean = batch[0].to(device), batch[1].to(device)
        valid_mask = batch[2].to(device) if len(batch) > 2 else None
        if noisy.ndim == 2:
            noisy = noisy.unsqueeze(1)
        if clean.ndim == 2:
            clean = clean.unsqueeze(1)

        batch_size = noisy.shape[0]
        t = torch.randint(0, self.timesteps, (batch_size,), device=device, dtype=torch.long)
        ecg_noise = noisy - clean
        gaussian_noise = torch.randn_like(clean)
        gamma, sigma = self._coefficients(t, clean.shape[-1])
        xt = clean + gamma * ecg_noise + sigma * gaussian_noise

        pred_ecg_noise, pred_gaussian_noise = self._predict(xt, noisy, t)
        pred_clean = self._clean_from_prediction(xt, t, pred_ecg_noise, pred_gaussian_noise)

        if valid_mask is not None:
            valid_mask = valid_mask.to(device, dtype=clean.dtype)
            denom = valid_mask.sum().clamp_min(1.0)

            def masked_mse(a, b):
                return ((a - b).pow(2) * valid_mask).sum() / denom

            ecg_loss = masked_mse(pred_ecg_noise, ecg_noise)
            gaussian_loss = masked_mse(pred_gaussian_noise, gaussian_noise)
            clean_loss = masked_mse(pred_clean, clean)
        else:
            ecg_loss = F.mse_loss(pred_ecg_noise, ecg_noise)
            gaussian_loss = F.mse_loss(pred_gaussian_noise, gaussian_noise)
            clean_loss = F.mse_loss(pred_clean, clean)

        pred_centered = pred_clean - pred_clean.mean(dim=-1, keepdim=True)
        clean_centered = clean - clean.mean(dim=-1, keepdim=True)
        pcc = (pred_centered * clean_centered).sum(dim=-1) / (
            pred_centered.norm(dim=-1) * clean_centered.norm(dim=-1) + 1.0e-8
        )
        pcc_loss = 1.0 - pcc.mean()

        return (
            self.ecg_noise_weight * ecg_loss
            + self.gaussian_noise_weight * gaussian_loss
            + self.clean_weight * clean_loss
            + self.pcc_weight * pcc_loss
        )
