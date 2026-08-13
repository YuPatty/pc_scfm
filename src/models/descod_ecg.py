import math

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .factory import register_model
except ImportError:
    from factory import register_model


class _KaimingConv1d(nn.Conv1d):
    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


class NoiseLevelEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = int(dim)

    def forward(self, noise_level):
        noise_level = noise_level.view(-1)
        count = self.dim // 2
        step = torch.arange(count, dtype=noise_level.dtype, device=noise_level.device) / count
        encoding = noise_level.unsqueeze(1) * torch.exp(-math.log(1.0e4) * step.unsqueeze(0))
        encoding = torch.cat([torch.sin(encoding), torch.cos(encoding)], dim=-1)
        if self.dim % 2:
            encoding = F.pad(encoding, (0, 1))
        return encoding


class FeatureWiseAffine(nn.Module):
    def __init__(self, in_channels, out_channels, use_affine_level=True):
        super().__init__()
        self.use_affine_level = bool(use_affine_level)
        self.noise_func = nn.Linear(
            int(in_channels),
            int(out_channels) * (1 + int(self.use_affine_level)),
        )

    def forward(self, x, noise_embed):
        batch = x.shape[0]
        if self.use_affine_level:
            gamma, beta = self.noise_func(noise_embed).view(batch, -1, 1).chunk(2, dim=1)
            return (1.0 + gamma) * x + beta
        return x + self.noise_func(noise_embed).view(batch, -1, 1)


class HNFBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        channels = int(channels)
        dilation = int(dilation)
        self.filters = nn.ModuleList(
            [
                _KaimingConv1d(channels, channels // 4, 3, dilation=dilation, padding=dilation, padding_mode="reflect"),
                _KaimingConv1d(channels, channels // 4, 5, dilation=dilation, padding=2 * dilation, padding_mode="reflect"),
                _KaimingConv1d(channels, channels // 4, 9, dilation=dilation, padding=4 * dilation, padding_mode="reflect"),
                _KaimingConv1d(channels, channels // 4, 15, dilation=dilation, padding=7 * dilation, padding_mode="reflect"),
            ]
        )
        self.conv_1 = _KaimingConv1d(channels, channels, 9, padding=4, padding_mode="reflect")
        self.norm = nn.InstanceNorm1d(channels // 2)
        self.conv_2 = _KaimingConv1d(channels, channels, 9, padding=4, padding_mode="reflect")

    def forward(self, x):
        residual = x
        features = torch.cat([layer(x) for layer in self.filters], dim=1)
        normalized, raw = self.conv_1(features).chunk(2, dim=1)
        features = F.leaky_relu(torch.cat([self.norm(normalized), raw], dim=1), 0.2)
        features = F.leaky_relu(self.conv_2(features), 0.2)
        return features + residual


class Bridge(nn.Module):
    def __init__(self, channels):
        super().__init__()
        channels = int(channels)
        self.input_conv = _KaimingConv1d(channels, channels, 3, padding=1, padding_mode="reflect")
        self.encoding = FeatureWiseAffine(channels, channels, use_affine_level=True)
        self.output_conv = _KaimingConv1d(channels, channels, 3, padding=1, padding_mode="reflect")

    def forward(self, x, noise_embed):
        x = self.input_conv(x)
        x = self.encoding(x, noise_embed)
        return self.output_conv(x)


class ConditionalScoreModel(nn.Module):
    def __init__(self, feats=80, dilations=(1, 2, 4, 2, 1)):
        super().__init__()
        feats = int(feats)
        self.stream_x_stem = nn.Sequential(
            _KaimingConv1d(1, feats, 9, padding=4, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
        )
        self.stream_cond_stem = nn.Sequential(
            _KaimingConv1d(1, feats, 9, padding=4, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
        )
        self.stream_x_blocks = nn.ModuleList([HNFBlock(feats, dilation) for dilation in dilations])
        self.stream_cond_blocks = nn.ModuleList([HNFBlock(feats, dilation) for dilation in dilations])
        self.embed = NoiseLevelEmbedding(feats)
        self.bridge = nn.ModuleList([Bridge(feats) for _ in dilations])
        self.conv_out = _KaimingConv1d(feats, 1, 9, padding=4, padding_mode="reflect")

    def forward(self, x, condition, noise_scale):
        noise_embed = self.embed(noise_scale)
        x = self.stream_x_stem(x)
        condition = self.stream_cond_stem(condition)
        for x_layer, condition_layer, bridge in zip(
            self.stream_x_blocks,
            self.stream_cond_blocks,
            self.bridge,
        ):
            x = x_layer(x)
            condition = condition_layer(condition) + bridge(x, noise_embed)
        return self.conv_out(condition)


class DeScoDDDPM(nn.Module):
    def __init__(
        self,
        feats=80,
        num_steps=50,
        beta_start=1.0e-4,
        beta_end=0.5,
        schedule="quad",
        loss_reduction="sum",
    ):
        super().__init__()
        self.num_steps = int(num_steps)
        self.loss_reduction = str(loss_reduction)
        self.model = ConditionalScoreModel(feats=int(feats))
        betas = self._make_beta_schedule(
            schedule=schedule,
            n_timesteps=self.num_steps,
            start=float(beta_start),
            end=float(beta_end),
        )
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_alphas_cumprod_prev",
            torch.sqrt(torch.cat([torch.ones(1), alphas_cumprod])),
        )
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1.0))

        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_log_variance_clipped", torch.log(posterior_variance.clamp_min(1.0e-20)))
        self.register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

    def _make_beta_schedule(self, schedule, n_timesteps, start, end):
        if schedule == "linear":
            return torch.linspace(start, end, n_timesteps)
        if schedule == "quad":
            return torch.linspace(start ** 0.5, end ** 0.5, n_timesteps) ** 2
        if schedule == "sigmoid":
            values = torch.linspace(-6, 6, n_timesteps)
            return torch.sigmoid(values) * (end - start) + start
        raise ValueError(f"Unsupported beta schedule: {schedule}")

    def predict_start_from_noise(self, x_t, t, noise):
        return self.sqrt_recip_alphas_cumprod[t] * x_t - self.sqrt_recipm1_alphas_cumprod[t] * noise

    def q_posterior(self, x_start, x_t, t):
        mean = self.posterior_mean_coef1[t] * x_start + self.posterior_mean_coef2[t] * x_t
        return mean, self.posterior_log_variance_clipped[t]

    def q_sample(self, x_start, continuous_sqrt_alpha_cumprod, noise):
        return continuous_sqrt_alpha_cumprod * x_start + (1.0 - continuous_sqrt_alpha_cumprod.pow(2)).sqrt() * noise

    def p_mean_variance(self, x, t, condition_x, clip_denoised=False):
        batch_size = x.shape[0]
        noise_level = self.sqrt_alphas_cumprod_prev[t + 1].view(1, 1).repeat(batch_size, 1)
        predicted_noise = self.model(x, condition_x, noise_level)
        x_recon = self.predict_start_from_noise(x, t=t, noise=predicted_noise)
        if clip_denoised:
            x_recon = x_recon.clamp(-1.0, 1.0)
        return self.q_posterior(x_start=x_recon, x_t=x, t=t)

    @torch.no_grad()
    def p_sample(self, x, t, condition_x, clip_denoised=False):
        model_mean, model_log_variance = self.p_mean_variance(
            x=x,
            t=t,
            condition_x=condition_x,
            clip_denoised=clip_denoised,
        )
        noise = torch.randn_like(x) if int(t.item()) > 0 else torch.zeros_like(x)
        return model_mean + noise * (0.5 * model_log_variance).exp()

    @torch.no_grad()
    def denoising_once(self, condition_x, clip_denoised=False):
        current = torch.randn_like(condition_x)
        for step in reversed(range(self.num_steps)):
            t = torch.full((1,), step, device=condition_x.device, dtype=torch.long)
            current = self.p_sample(current, t, condition_x=condition_x, clip_denoised=clip_denoised)
        return current

    def p_losses(self, clean, noisy, valid_mask=None, noise=None):
        batch_size = clean.shape[0]
        step = torch.randint(1, self.num_steps + 1, (1,), device=clean.device).item()
        lower = self.sqrt_alphas_cumprod_prev[step - 1]
        upper = self.sqrt_alphas_cumprod_prev[step]
        continuous = lower + (upper - lower) * torch.rand(batch_size, 1, device=clean.device)
        noise = torch.randn_like(clean) if noise is None else noise
        x_noisy = self.q_sample(clean, continuous.view(-1, 1, 1), noise)
        predicted_noise = self.model(x_noisy, noisy, continuous)

        loss = F.l1_loss(predicted_noise, noise, reduction="none")
        if valid_mask is not None:
            if valid_mask.ndim == 2:
                valid_mask = valid_mask.unsqueeze(1)
            valid_mask = valid_mask.to(device=loss.device, dtype=loss.dtype)
            return (loss * valid_mask).sum() / valid_mask.sum().clamp_min(1.0)
        if self.loss_reduction == "sum":
            return loss.sum()
        if self.loss_reduction == "mean":
            return loss.mean()
        raise ValueError(f"Unsupported loss_reduction: {self.loss_reduction}")


@register_model("descod_ecg_1shot")
@register_model("descod_ecg_5shot")
@register_model("descod_ecg_10shot")
class DeScoDECGDenoiser(nn.Module):
    def __init__(
        self,
        feats=80,
        num_steps=50,
        beta_start=1.0e-4,
        beta_end=0.5,
        schedule="quad",
        num_shots=1,
        clip_denoised=False,
        loss_reduction="sum",
        loss_fn="noise_l1",
        **kwargs,
    ):
        super().__init__()
        if loss_fn != "noise_l1":
            raise ValueError("DeScoDECGDenoiser currently supports only model.loss_fn='noise_l1'.")
        self.num_shots = int(num_shots)
        self.clip_denoised = bool(clip_denoised)
        self.diffusion = DeScoDDDPM(
            feats=int(feats),
            num_steps=int(num_steps),
            beta_start=float(beta_start),
            beta_end=float(beta_end),
            schedule=str(schedule),
            loss_reduction=str(loss_reduction),
        )

    def forward(self, x):
        squeeze = False
        if x.ndim == 2:
            x = x.unsqueeze(1)
            squeeze = True
        if x.ndim != 3 or x.shape[1] != 1:
            raise ValueError(f"Expected ECG shaped [B, T] or [B, 1, T], got {tuple(x.shape)}.")

        samples = [
            self.diffusion.denoising_once(x, clip_denoised=self.clip_denoised)
            for _ in range(max(self.num_shots, 1))
        ]
        restored = torch.stack(samples, dim=0).mean(dim=0)
        return restored.squeeze(1) if squeeze else restored

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
        return self.diffusion.p_losses(clean=clean, noisy=noisy, valid_mask=valid_mask)
