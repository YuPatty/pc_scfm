from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .factory import register_model
except ImportError:
    from factory import register_model


def _load_mamba_symbols():
    try:
        from mamba_ssm.modules.mamba_simple import Mamba
        from mamba_ssm.modules.block import Block
        from mamba_ssm.models.mixer_seq_simple import _init_weights
        try:
            from mamba_ssm.ops.triton.layer_norm import RMSNorm
        except ImportError:
            from mamba_ssm.ops.triton.layernorm import RMSNorm
    except ImportError as exc:
        raise ImportError(
            "MECG-E/MambAttention models require `mamba_ssm`. Install the Mamba SSM "
            "dependency before instantiating model_name=mecg_e or model_name=mambattention_ecg."
        ) from exc
    return Mamba, Block, _init_weights, RMSNorm


def get_padding_2d(kernel_size, dilation=(1, 1)):
    return (
        int((kernel_size[0] * dilation[0] - dilation[0]) / 2),
        int((kernel_size[1] * dilation[1] - dilation[1]) / 2),
    )


def mag_pha_stft(y, n_fft, hop_size, win_size, compress_factor=1.0, center=True):
    hann_window = torch.hann_window(win_size, device=y.device)
    stft_spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window,
        center=center,
        pad_mode="reflect",
        normalized=False,
        return_complex=True,
    )
    mag = torch.abs(stft_spec)
    pha = torch.angle(stft_spec)
    mag = torch.pow(mag, compress_factor)
    com = torch.stack((mag * torch.cos(pha), mag * torch.sin(pha)), dim=-1)
    return mag, pha, com


def mag_pha_stft_loss(y, n_fft, hop_size, win_size, compress_factor=1.0, center=True):
    hann_window = torch.hann_window(win_size, device=y.device)
    stft_spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window,
        center=center,
        pad_mode="reflect",
        normalized=False,
        return_complex=True,
    )
    stft_spec = torch.stack((stft_spec.real, stft_spec.imag), dim=-1)
    mag = torch.sqrt(stft_spec.pow(2).sum(-1) + 1e-9)
    pha = torch.atan2(stft_spec[..., 1] + 1e-10, stft_spec[..., 0] + 1e-5)
    mag = torch.pow(mag, compress_factor)
    com = torch.stack((mag * torch.cos(pha), mag * torch.sin(pha)), dim=-1)
    return mag, pha, com


def mag_pha_istft(mag, pha, n_fft, hop_size, win_size, compress_factor=1.0, center=True):
    mag = torch.pow(mag, 1.0 / compress_factor)
    com = torch.complex(mag * torch.cos(pha), mag * torch.sin(pha))
    hann_window = torch.hann_window(win_size, device=com.device)
    return torch.istft(
        com,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window,
        center=center,
    )


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


class LearnableSigmoid2d(nn.Module):
    def __init__(self, in_features, beta=1):
        super().__init__()
        self.beta = beta
        self.slope = nn.Parameter(torch.ones(in_features, 1))

    def forward(self, x):
        return self.beta * torch.sigmoid(self.slope * x)


class MambaBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        n_layer=1,
        bidirectional=False,
        d_state=16,
        d_conv=4,
        expand=4,
        norm_epsilon=1e-5,
    ):
        super().__init__()
        Mamba, Block, init_weights, RMSNorm = _load_mamba_symbols()
        self.bidirectional = bidirectional
        self.forward_blocks = nn.ModuleList(
            [
                Block(
                    in_channels,
                    mixer_cls=partial(
                        Mamba,
                        layer_idx=i,
                        d_state=d_state,
                        d_conv=d_conv,
                        expand=expand,
                        use_fast_path=True,
                    ),
                    mlp_cls=nn.Identity,
                    norm_cls=partial(RMSNorm, eps=norm_epsilon),
                    fused_add_norm=False,
                )
                for i in range(n_layer)
            ]
        )
        self.backward_blocks = nn.ModuleList()
        if bidirectional:
            self.backward_blocks = nn.ModuleList(
                [
                    Block(
                        in_channels,
                        mixer_cls=partial(
                            Mamba,
                            layer_idx=i,
                            d_state=d_state,
                            d_conv=d_conv,
                            expand=expand,
                            use_fast_path=True,
                        ),
                        mlp_cls=nn.Identity,
                        norm_cls=partial(RMSNorm, eps=norm_epsilon),
                        fused_add_norm=False,
                    )
                    for i in range(n_layer)
                ]
            )
        self.apply(partial(init_weights, n_layer=n_layer))

    def forward(self, input):
        forward_residual = None
        forward_f = input.clone()
        for block in self.forward_blocks:
            forward_f, forward_residual = block(forward_f, forward_residual, inference_params=None)
        residual = forward_f + forward_residual if forward_residual is not None else forward_f

        if self.bidirectional:
            backward_residual = None
            backward_f = torch.flip(input, [1])
            for block in self.backward_blocks:
                backward_f, backward_residual = block(
                    backward_f, backward_residual, inference_params=None
                )
            backward_residual = (
                backward_f + backward_residual if backward_residual is not None else backward_f
            )
            residual = torch.cat([residual, torch.flip(backward_residual, [1])], -1)
        return residual


class DenseBlock(nn.Module):
    def __init__(self, h, kernel_size=(3, 3), depth=4):
        super().__init__()
        self.dense_block = nn.ModuleList()
        for i in range(depth):
            dil = 2**i
            self.dense_block.append(
                nn.Sequential(
                    nn.Conv2d(
                        h.dense_channel * (i + 1),
                        h.dense_channel,
                        kernel_size,
                        dilation=(dil, 1),
                        padding=get_padding_2d(kernel_size, (dil, 1)),
                    ),
                    nn.InstanceNorm2d(h.dense_channel, affine=True),
                    nn.PReLU(h.dense_channel),
                )
            )

    def forward(self, x):
        skip = x
        for dense_conv in self.dense_block:
            x = dense_conv(skip)
            skip = torch.cat([x, skip], dim=1)
        return x


class DenseEncoder(nn.Module):
    def __init__(self, h, in_channel):
        super().__init__()
        self.dense_conv_1 = nn.Sequential(
            nn.Conv2d(in_channel, h.dense_channel, (1, 1)),
            nn.InstanceNorm2d(h.dense_channel, affine=True),
            nn.PReLU(h.dense_channel),
        )
        self.dense_block = DenseBlock(h, depth=h.get("edepth", 4))
        self.dense_conv_2 = nn.Sequential(
            nn.Conv2d(h.dense_channel, h.dense_channel, (1, 3), (1, 2)),
            nn.InstanceNorm2d(h.dense_channel, affine=True),
            nn.PReLU(h.dense_channel),
        )

    def forward(self, x):
        x = self.dense_conv_1(x)
        x = self.dense_block(x)
        return self.dense_conv_2(x)


class MaskDecoder(nn.Module):
    def __init__(self, h, out_channel=1):
        super().__init__()
        self.dense_block = DenseBlock(h, depth=h.get("mdepth", 4))
        self.mask_conv = nn.Sequential(
            nn.ConvTranspose2d(h.dense_channel, h.dense_channel, (1, 3), (1, 2)),
            nn.Conv2d(h.dense_channel, out_channel, (1, 1)),
            nn.InstanceNorm2d(out_channel, affine=True),
            nn.PReLU(out_channel),
            nn.Conv2d(out_channel, out_channel, (1, 1)),
        )
        self.lsigmoid = LearnableSigmoid2d(h.n_fft // 2 + 1, beta=h.beta)

    def forward(self, x):
        x = self.dense_block(x)
        x = self.mask_conv(x)
        x = x.permute(0, 3, 2, 1).squeeze(-1)
        return self.lsigmoid(x).permute(0, 2, 1).unsqueeze(1)


class PhaseDecoder(nn.Module):
    def __init__(self, h, out_channel=1):
        super().__init__()
        self.dense_block = DenseBlock(h, depth=h.get("pdepth", 4))
        self.phase_conv = nn.Sequential(
            nn.ConvTranspose2d(h.dense_channel, h.dense_channel, (1, 3), (1, 2)),
            nn.InstanceNorm2d(h.dense_channel, affine=True),
            nn.PReLU(h.dense_channel),
        )
        self.phase_conv_r = nn.Conv2d(h.dense_channel, out_channel, (1, 1))
        self.phase_conv_i = nn.Conv2d(h.dense_channel, out_channel, (1, 1))

    def forward(self, x):
        x = self.dense_block(x)
        x = self.phase_conv(x)
        return torch.atan2(self.phase_conv_i(x), self.phase_conv_r(x))


class ComplexDecoder(nn.Module):
    def __init__(self, h, out_channel=1):
        super().__init__()
        self.dense_block = DenseBlock(h, depth=h.get("pdepth", 4))
        self.phase_conv = nn.Sequential(
            nn.ConvTranspose2d(h.dense_channel, h.dense_channel, (1, 3), (1, 2)),
            nn.InstanceNorm2d(h.dense_channel, affine=True),
            nn.PReLU(h.dense_channel),
        )
        self.phase_conv_r = nn.Conv2d(h.dense_channel, out_channel, (1, 1))
        self.phase_conv_i = nn.Conv2d(h.dense_channel, out_channel, (1, 1))

    def forward(self, x):
        x = self.dense_block(x)
        x = self.phase_conv(x)
        return torch.cat((self.phase_conv_r(x), self.phase_conv_i(x)), dim=1)


class TSMambaBlock(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.h = h
        self.time_mamba = MambaBlock(
            h.dense_channel,
            n_layer=1,
            bidirectional=True,
            d_state=h.get("d_state", 16),
            d_conv=h.get("d_conv", 4),
            expand=h.get("expand", 4),
            norm_epsilon=h.get("norm_epsilon", 1e-5),
        )
        self.freq_mamba = MambaBlock(
            h.dense_channel,
            n_layer=1,
            bidirectional=True,
            d_state=h.get("d_state", 16),
            d_conv=h.get("d_conv", 4),
            expand=h.get("expand", 4),
            norm_epsilon=h.get("norm_epsilon", 1e-5),
        )
        self.tlinear = nn.ConvTranspose1d(h.dense_channel * 2, h.dense_channel, 1)
        self.flinear = nn.ConvTranspose1d(h.dense_channel * 2, h.dense_channel, 1)

    def forward(self, x):
        b, c, t, f = x.size()
        x = x.permute(0, 3, 2, 1).contiguous().view(b * f, t, c)
        x = self.tlinear(self.time_mamba(x).permute(0, 2, 1)).permute(0, 2, 1) + x
        x = x.view(b, f, t, c)
        if self.h.get("fmamba", True):
            x = x.permute(0, 2, 1, 3).contiguous().view(b * t, f, c)
            x = self.flinear(self.freq_mamba(x).permute(0, 2, 1)).permute(0, 2, 1) + x
            x = x.view(b, t, f, c).permute(0, 2, 1, 3)
        return x.permute(0, 3, 2, 1)

try:
    from .pc_scfm_components import FlowBaselineHead, MorphologySafetyLayer, RiskPolicyHead
except ImportError:
    from pc_scfm_components import FlowBaselineHead, MorphologySafetyLayer, RiskPolicyHead

class MECGECore(nn.Module):
    def __init__(self, config, block_cls=TSMambaBlock):
        super().__init__()
        h = AttrDict(config["model"])
        self.h = h
        self.fea = h.get("fea", "pha")
        self.norm = h.get("norm", False)
        self.loss_fn = h.get("loss_fn", "time+com+con").split("+")
        self.num_tscblocks = h.num_tscblocks
        self.pcscfm_enabled = h.get("pcscfm_enabled", False)
        self.t_max = h.get("t_max", 1)
        self.stop_threshold = h.get("stop_threshold", 0.95)
        self.reject_threshold = h.get("reject_threshold", 0.98)
        self.classical_kernel_size = h.get("classical_kernel_size", 129)
        self.baseline_kernel_size = h.get("baseline_kernel_size", 129)
        self.flow_nfe = h.get("flow_nfe", 4)
        self.flow_samples = h.get("flow_samples", 4)
        self.lambda_bc = h.get("lambda_bc", 0.02)
        self.lambda_policy_value = h.get("lambda_policy_value", 0.01)
        self.lambda_step = h.get("lambda_step", 0.001)
        self.lambda_reject = h.get("lambda_reject", 0.01)
        self.reject_uncertainty_threshold = h.get("reject_uncertainty_threshold", 0.25)
        self.reject_disagreement_threshold = h.get("reject_disagreement_threshold", 0.25)
        self.stochastic_policy = h.get("stochastic_policy", False)
        self.policy_mode = h.get("policy_mode", "learned")
        self.phase_representation = h.get("phase_representation", "raw")
        self.last_metadata = None

        in_channel = 3 if self.phase_representation == "sincos" and self.fea == "pha" else 2
        self.dense_encoder = DenseEncoder(h, in_channel=in_channel)
        self.tsc_blocks = nn.ModuleList([block_cls(h) for _ in range(h.num_tscblocks)])
        self.mask_decoder = MaskDecoder(h, out_channel=1)

        if self.fea == "cpx":
            self.complex_decoder = ComplexDecoder(h, out_channel=1)
        elif self.fea == "wav":
            self.encoder = nn.Conv1d(
                1, (h.n_fft // 2 + 1) * 2, h.win_size, h.hop_size, padding=h.win_size // 2
            )
            self.decoder = nn.ConvTranspose1d(
                (h.n_fft // 2 + 1) * 2,
                1,
                h.win_size,
                h.hop_size,
                padding=h.win_size // 2,
                output_padding=0,
            )
            self.complex_decoder = ComplexDecoder(h, out_channel=1)
        elif self.fea == "pha":
            self.phase_decoder = PhaseDecoder(h, out_channel=1)
        else:
            raise NotImplementedError(f"Feature '{self.fea}' is not implemented.")

        if self.pcscfm_enabled:
            self.flow_head = FlowBaselineHead(h)
            self.policy_head = RiskPolicyHead(h, state_dim=h.dense_channel + 15)
            self.safety_layer = MorphologySafetyLayer(h)

    def _norm_factor(self, noisy):
        if self.norm == "1":
            return torch.sqrt(noisy.shape[-1] / torch.sum(noisy**2.0, -1, keepdim=True))
        if self.norm == "2":
            return 1 / noisy.abs().max(-1, keepdim=True)[0]
        return torch.ones((noisy.shape[0], 1, 1), device=noisy.device)

    def _encode_noisy(self, noisy_audio):
        noisy_mag, noisy_pha, noisy_com = mag_pha_stft(
            noisy_audio,
            self.h.n_fft,
            self.h.hop_size,
            self.h.win_size,
            self.h.compress_factor,
        )
        noisy_mag_4d = noisy_mag.unsqueeze(-1).permute(0, 3, 2, 1)

        if self.fea == "cpx":
            x = noisy_com.permute(0, 3, 2, 1)
        elif self.fea == "pha":
            if self.phase_representation == "sincos":
                noisy_sin = torch.sin(noisy_pha).unsqueeze(-1).permute(0, 3, 2, 1)
                noisy_cos = torch.cos(noisy_pha).unsqueeze(-1).permute(0, 3, 2, 1)
                x = torch.cat((noisy_mag_4d, noisy_sin, noisy_cos), dim=1)
            else:
                noisy_pha_4d = noisy_pha.unsqueeze(-1).permute(0, 3, 2, 1)
                x = torch.cat((noisy_mag_4d, noisy_pha_4d), dim=1)
        elif self.fea == "wav":
            x = self.encoder(noisy_audio.unsqueeze(1))
            b, channels, frames = x.shape
            x = x.view(b, 2, -1, frames).permute(0, 1, 3, 2)
        else:
            raise NotImplementedError(f"Feature '{self.fea}' is not implemented.")

        x = self.dense_encoder(x)
        for block in self.tsc_blocks:
            x = block(x)
        return x, noisy_mag_4d, noisy_pha

    def _smooth_1d(self, x, kernel_size):
        if kernel_size <= 1:
            return x
        kernel_size = min(kernel_size, x.shape[-1] if x.shape[-1] % 2 == 1 else x.shape[-1] - 1)
        kernel_size = max(kernel_size, 1)
        padding = kernel_size // 2
        return F.avg_pool1d(x, kernel_size=kernel_size, stride=1, padding=padding, count_include_pad=False)

    def _baseline_projection(self, x):
        return self._smooth_1d(x, self.baseline_kernel_size)

    def _classical_baseline(self, x):
        return self._smooth_1d(x, self.classical_kernel_size)

    def _validity_features(self, x, valid_mask=None):
        finite = torch.isfinite(x).float()
        if valid_mask is not None:
            finite = finite * valid_mask.float()
        x_safe = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0) * finite
        peak = x_safe.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6)
        clipping_mask = (x_safe.abs() > 0.98 * peak).float()
        return {
            "valid_mask": finite,
            "clipping_ratio": clipping_mask.mean(dim=-1),
            "missing_ratio": (1.0 - finite).mean(dim=-1),
            "x_safe": x_safe,
        }

    def _baseline_coefficients(self, baseline):
        coeff_count = self.h.get("flow_coeff_count", 32)
        if self.h.get("flow_basis", "cosine") != "cosine":
            return F.interpolate(baseline, size=coeff_count, mode="linear", align_corners=False).squeeze(1)
        signal_len = baseline.shape[-1]
        t = torch.linspace(0.0, 1.0, signal_len, device=baseline.device, dtype=baseline.dtype)
        basis = torch.stack([torch.cos(torch.pi * k * t) for k in range(coeff_count)], dim=0)
        denom = basis.pow(2).sum(dim=-1).clamp_min(1e-6)
        return torch.matmul(baseline.squeeze(1), basis.t()) / denom

    def restore_one_shot(self, noisy_audio, return_com=False):
        if noisy_audio.ndim == 3:
            noisy_audio = noisy_audio.squeeze(1)
        x, noisy_mag_4d, noisy_pha = self._encode_noisy(noisy_audio)
        mag_g = (noisy_mag_4d * self.mask_decoder(x)).permute(0, 3, 2, 1).squeeze(-1)

        if self.fea == "cpx":
            com_d = self.complex_decoder(x).permute(0, 3, 2, 1)
            com_g = torch.stack(
                (mag_g * torch.cos(noisy_pha), mag_g * torch.sin(noisy_pha)), dim=-1
            )
            pha_g = torch.angle(torch.complex((com_g + com_d)[..., 0], (com_g + com_d)[..., 1]))
            restored = mag_pha_istft(
                mag_g, pha_g, self.h.n_fft, self.h.hop_size, self.h.win_size, self.h.compress_factor
            )
        elif self.fea == "pha":
            pha_g = self.phase_decoder(x).permute(0, 3, 2, 1).squeeze(-1)
            restored = mag_pha_istft(
                mag_g, pha_g, self.h.n_fft, self.h.hop_size, self.h.win_size, self.h.compress_factor
            )
        else:
            b, channels, frames = self.encoder(noisy_audio.unsqueeze(1)).shape
            com_d = self.complex_decoder(x).permute(0, 1, 3, 2).reshape(b, channels, frames)
            restored = self.decoder(com_d).squeeze(1)
            _, _, com_g = mag_pha_stft_loss(
                restored, self.h.n_fft, self.h.hop_size, self.h.win_size, self.h.compress_factor
            )
        output = restored.unsqueeze(1)
        if return_com:
            return output, com_g
        return output

    def restore(self, noisy_audio):
        restored, _ = self.restore_with_metadata(noisy_audio)
        return restored

    def restore_with_metadata(self, noisy_audio, valid_mask=None):
        if noisy_audio.ndim == 2:
            noisy_audio = noisy_audio.unsqueeze(1)
        if noisy_audio.shape[1] != 1:
            raise ValueError(f"MECG-E expects single-lead input shaped [B, 1, T], got {noisy_audio.shape}.")

        norm_factor = self._norm_factor(noisy_audio)
        noisy_audio_norm = (noisy_audio * norm_factor).squeeze(1)
        if self.pcscfm_enabled:
            if valid_mask is not None:
                valid_mask = valid_mask.to(noisy_audio.device)
            restored, _, _, metadata = self._pcscfm_restoration(
                noisy_audio_norm.unsqueeze(1),
                valid_mask=valid_mask,
                training=False,
                return_metadata=True,
            )
            self.last_metadata = metadata
            return restored.unsqueeze(1) / norm_factor, metadata

        restored = self.restore_one_shot(noisy_audio_norm)
        self.last_metadata = None
        return restored / norm_factor, None

    def _belief_state(
        self,
        pooled_feature,
        current,
        original,
        direct_baseline,
        flow_mu,
        flow_logvar,
        classical_baseline,
        validity,
        step_idx,
    ):
        flow_var = flow_logvar.exp()
        cumulative = original - current
        proposal_disagreement = torch.stack(
            [
                (direct_baseline - flow_mu).abs().mean(dim=-1),
                (direct_baseline - classical_baseline).abs().mean(dim=-1),
                (flow_mu - classical_baseline).abs().mean(dim=-1),
            ],
            dim=0,
        ).mean(dim=0)
        stats = torch.cat(
            [
                direct_baseline.abs().mean(dim=-1),
                flow_mu.abs().mean(dim=-1),
                flow_var.mean(dim=-1),
                classical_baseline.abs().mean(dim=-1),
                self._baseline_projection(current).abs().mean(dim=-1),
                cumulative.abs().mean(dim=-1),
                current.std(dim=-1),
                current.amax(dim=-1) - current.amin(dim=-1),
                validity["clipping_ratio"],
                validity["missing_ratio"],
                proposal_disagreement,
                (direct_baseline * flow_mu).mean(dim=-1),
                current.abs().median(dim=-1).values,
                current.mean(dim=-1),
                torch.full_like(current.mean(dim=-1), float(step_idx) / max(self.t_max, 1)),
            ],
            dim=-1,
        )
        return torch.cat([pooled_feature, stats], dim=-1)

    def _oracle_action(self, current, clean_audio, direct_baseline, flow_mu, classical_baseline):
        residual = current - clean_audio.unsqueeze(1)
        proposals = torch.cat([direct_baseline, flow_mu, classical_baseline], dim=1)
        scores = (proposals * residual).mean(dim=-1)
        best = scores.argmax(dim=1)
        weights = F.one_hot(best, num_classes=3).float().unsqueeze(-1)
        chosen = (proposals * weights).sum(dim=1, keepdim=True)
        numerator = (residual * chosen).sum(dim=-1, keepdim=True)
        denominator = (chosen**2).sum(dim=-1, keepdim=True).clamp_min(1e-6)
        alpha = (numerator / denominator).clamp(0.0, 1.0)
        current_loss = (current.squeeze(1) - clean_audio).abs().mean(dim=-1, keepdim=True)
        stop = (current_loss < self.h.get("oracle_stop_l1", 0.02)).float().unsqueeze(-1)
        reject = (scores.max(dim=1, keepdim=True).values < 0).float().unsqueeze(-1)
        return alpha, weights, stop, reject

    def _pcscfm_restoration(self, noisy_audio, clean_audio=None, valid_mask=None, training=False, return_metadata=False):
        validity = self._validity_features(noisy_audio, valid_mask=valid_mask)
        current = validity["x_safe"]
        original = current
        cumulative_cost = torch.zeros((noisy_audio.shape[0], 1, 1), device=noisy_audio.device)
        history = []
        last_com = None

        for step_idx in range(self.t_max):
            direct_clean = self.restore_one_shot(current)
            direct_baseline = current - direct_clean
            encoded, _, _ = self._encode_noisy(current.squeeze(1))
            _, _, last_com = mag_pha_stft_loss(
                direct_clean.squeeze(1),
                self.h.n_fft,
                self.h.hop_size,
                self.h.win_size,
                self.h.compress_factor,
            )
            flow_mu, flow_logvar, pooled_feature, flow_mu_coeff = self.flow_head(
                encoded,
                current.shape[-1],
                nfe=self.flow_nfe,
                samples=self.flow_samples,
            )
            classical_baseline = self._classical_baseline(current)
            state = self._belief_state(
                pooled_feature,
                current,
                original,
                direct_baseline,
                flow_mu,
                flow_logvar,
                classical_baseline,
                validity,
                step_idx,
            )
            if self.policy_mode == "fixed_multistep":
                batch = state.shape[0]
                weights = torch.zeros((batch, 3, 1), device=state.device, dtype=state.dtype)
                weights[:, 0:1] = 1.0
                action = {
                    "alpha": torch.full((batch, 1, 1), 1.0 / max(self.t_max, 1), device=state.device, dtype=state.dtype),
                    "weights": weights,
                    "stop_prob": torch.zeros((batch, 1, 1), device=state.device, dtype=state.dtype),
                    "reject_prob": torch.zeros((batch, 1, 1), device=state.device, dtype=state.dtype),
                    "value": torch.zeros((batch, 1, 1), device=state.device, dtype=state.dtype),
                    "cost_value": torch.zeros((batch, 1, 1), device=state.device, dtype=state.dtype),
                    "weight_index": torch.zeros((batch, 1), device=state.device, dtype=torch.long),
                    "stop_sample": torch.zeros((batch, 1, 1), device=state.device, dtype=state.dtype),
                    "reject_sample": torch.zeros((batch, 1, 1), device=state.device, dtype=state.dtype),
                    "log_prob": torch.zeros((batch, 1, 1), device=state.device, dtype=state.dtype),
                    "entropy": torch.zeros((batch, 1, 1), device=state.device, dtype=state.dtype),
                }
            else:
                action = self.policy_head.sample_action(
                    state,
                    deterministic=(not training or not self.stochastic_policy),
                )
            weights = action["weights"]
            mixed_baseline = (
                weights[:, 0:1] * direct_baseline
                + weights[:, 1:2] * flow_mu
                + weights[:, 2:3] * classical_baseline
            )
            delta = self._baseline_projection(mixed_baseline)
            alpha_safe, cumulative_cost, morph_cost = self.safety_layer(action["alpha"], delta, cumulative_cost)
            next_current = current - alpha_safe * delta

            item = {
                "direct_clean": direct_clean,
                "direct_baseline": direct_baseline,
                "flow_mu": flow_mu,
                "flow_logvar": flow_logvar,
                "flow_mu_coeff": flow_mu_coeff,
                "classical_baseline": classical_baseline,
                "alpha": action["alpha"],
                "alpha_safe": alpha_safe,
                "weights": weights,
                "stop_logit": action["stop_logit"],
                "reject_logit": action["reject_logit"],
                "stop_prob": action["stop_prob"],
                "reject_prob": action["reject_prob"],
                "weight_index": action.get("weight_index", action["weights"].squeeze(-1).argmax(dim=-1, keepdim=True)),
                "stop_sample": action.get("stop_sample", (action["stop_prob"] >= 0.5).float()),
                "reject_sample": action.get("reject_sample", (action["reject_prob"] >= 0.5).float()),
                "log_prob": action.get("log_prob", torch.zeros_like(action["alpha"])),
                "entropy": action.get("entropy", torch.zeros_like(action["alpha"])),
                "value": action["value"],
                "cost_value": action["cost_value"],
                "morph_cost": morph_cost,
                "state": state,
            }
            if clean_audio is not None:
                oracle_alpha, oracle_weights, oracle_stop, oracle_reject = self._oracle_action(
                    current,
                    clean_audio,
                    direct_baseline,
                    flow_mu,
                    classical_baseline,
                )
                item.update(
                    {
                        "oracle_alpha": oracle_alpha,
                        "oracle_weights": oracle_weights,
                        "oracle_stop": oracle_stop,
                        "oracle_reject": oracle_reject,
                        "flow_target_coeff": self._baseline_coefficients(
                            self._baseline_projection(current - clean_audio.unsqueeze(1))
                        ),
                        "encoded": encoded,
                    }
                )
            history.append(item)

            current = next_current
            if not training:
                should_stop = (action["stop_prob"] >= self.stop_threshold).view(-1, 1, 1)
                flow_uncertainty = flow_logvar.exp().mean(dim=-1, keepdim=True)
                disagreement = torch.stack(
                    [
                        (direct_baseline - flow_mu).abs().mean(dim=-1, keepdim=True),
                        (direct_baseline - classical_baseline).abs().mean(dim=-1, keepdim=True),
                        (flow_mu - classical_baseline).abs().mean(dim=-1, keepdim=True),
                    ],
                    dim=0,
                ).mean(dim=0)
                should_reject = (
                    (action["reject_prob"] >= self.reject_threshold)
                    | (flow_uncertainty >= self.reject_uncertainty_threshold)
                    | (disagreement >= self.reject_disagreement_threshold)
                    | (validity["clipping_ratio"].unsqueeze(-1) > 0.05)
                    | (validity["missing_ratio"].unsqueeze(-1) > 0.0)
                )
                reject_reason = torch.zeros_like(should_reject, dtype=torch.long)
                reject_reason = torch.where(
                    flow_uncertainty >= self.reject_uncertainty_threshold,
                    torch.full_like(reject_reason, 1),
                    reject_reason,
                )
                reject_reason = torch.where(
                    disagreement >= self.reject_disagreement_threshold,
                    torch.full_like(reject_reason, 2),
                    reject_reason,
                )
                reject_reason = torch.where(
                    validity["clipping_ratio"].unsqueeze(-1) > 0.05,
                    torch.full_like(reject_reason, 3),
                    reject_reason,
                )
                reject_reason = torch.where(
                    validity["missing_ratio"].unsqueeze(-1) > 0.0,
                    torch.full_like(reject_reason, 4),
                    reject_reason,
                )
                reject_reason = torch.where(
                    action["reject_prob"] >= self.reject_threshold,
                    torch.full_like(reject_reason, 5),
                    reject_reason,
                )
                history[-1]["hard_stop"] = should_stop.float()
                history[-1]["hard_reject"] = should_reject.float()
                history[-1]["reject_reason"] = reject_reason
                if (should_stop | should_reject).all():
                    break

        metadata = self._build_metadata(history, validity)
        if return_metadata:
            return current.squeeze(1), last_com, history, metadata
        return current.squeeze(1), last_com, history

    def _build_metadata(self, history, validity):
        if not history:
            return {}
        alpha = torch.cat([item["alpha"].detach().view(-1, 1) for item in history], dim=1)
        alpha_safe = torch.cat([item["alpha_safe"].detach().view(-1, 1) for item in history], dim=1)
        stop_prob = torch.cat([item["stop_prob"].detach().view(-1, 1) for item in history], dim=1)
        reject_prob = torch.cat([item["reject_prob"].detach().view(-1, 1) for item in history], dim=1)
        weights = torch.stack([item["weights"].detach().squeeze(-1) for item in history], dim=1)
        flow_uncertainty = torch.cat([item["flow_logvar"].detach().exp().mean(dim=-1) for item in history], dim=1)
        hard_reject = history[-1].get(
            "hard_reject", reject_prob[:, -1:].unsqueeze(-1) >= self.reject_threshold
        ).detach().view(-1, 1)
        hard_stop = history[-1].get(
            "hard_stop", stop_prob[:, -1:].unsqueeze(-1) >= self.stop_threshold
        ).detach().view(-1, 1)
        reject_reason = history[-1].get(
            "reject_reason", torch.zeros_like(hard_reject, dtype=torch.long)
        ).detach().view(-1, 1)
        return {
            "steps": torch.full_like(hard_reject, len(history), dtype=torch.float32),
            "alpha": alpha,
            "alpha_safe": alpha_safe,
            "proposal_weights": weights,
            "stop_prob": stop_prob,
            "reject_prob": reject_prob,
            "hard_stop": hard_stop.float(),
            "hard_reject": hard_reject.float(),
            "coverage": 1.0 - hard_reject.float(),
            "flow_uncertainty": flow_uncertainty,
            "reject_reason": reject_reason,
            "clipping_ratio": validity["clipping_ratio"].detach(),
            "missing_ratio": validity["missing_ratio"].detach(),
        }

    def _masked_mean(self, value, valid_mask=None):
        if valid_mask is None:
            return value.mean()
        mask = valid_mask.squeeze(1) if valid_mask.dim() == 3 else valid_mask
        while mask.dim() < value.dim():
            mask = mask.unsqueeze(-1)
        mask = mask.to(value.device, dtype=value.dtype)
        return (value * mask).sum() / mask.sum().clamp_min(1.0)

    def _ecg_loss(self, clean_audio, restored_audio, norm_factor, predicted_com=None, aux_history=None, valid_mask=None):
        _, _, clean_com = mag_pha_stft(
            clean_audio, self.h.n_fft, self.h.hop_size, self.h.win_size, self.h.compress_factor
        )
        _, _, restored_com = mag_pha_stft_loss(
            restored_audio, self.h.n_fft, self.h.hop_size, self.h.win_size, self.h.compress_factor
        )
        com_g = predicted_com if predicted_com is not None else restored_com
        loss = clean_audio.new_tensor(0.0)

        if "time" in self.loss_fn:
            loss_time = F.l1_loss(clean_audio, restored_audio, reduction="none")
            loss = loss + 0.5 * self._masked_mean(loss_time / norm_factor.squeeze(-1), valid_mask)
        if "com" in self.loss_fn:
            loss_com = F.mse_loss(clean_com, restored_com, reduction="none") * 2
            loss = loss + 0.5 * (loss_com / norm_factor.unsqueeze(-1)).mean()
        if "con" in self.loss_fn:
            loss_con = F.mse_loss(com_g, restored_com, reduction="none") * 2
            loss = loss + 0.5 * (loss_con / norm_factor.unsqueeze(-1)).mean()
        if "lf" in self.loss_fn:
            lf_residual = self._baseline_projection((restored_audio - clean_audio).unsqueeze(1)).squeeze(1)
            loss = loss + self._masked_mean(lf_residual.abs(), valid_mask) * self.h.get("lambda_lf", 0.2)
        if "morph" in self.loss_fn:
            clean_derivative = clean_audio[..., 1:] - clean_audio[..., :-1]
            restored_derivative = restored_audio[..., 1:] - restored_audio[..., :-1]
            derivative_loss = F.l1_loss(restored_derivative, clean_derivative, reduction="none")
            derivative_mask = valid_mask[..., 1:] * valid_mask[..., :-1] if valid_mask is not None else None
            loss = loss + self._masked_mean(derivative_loss, derivative_mask) * self.h.get("lambda_morph", 0.1)

        if aux_history and "flow" in self.loss_fn:
            flow_loss = clean_audio.new_tensor(0.0)
            for item in aux_history:
                if "flow_target_coeff" in item and "encoded" in item:
                    flow_loss = flow_loss + self.flow_head.flow_matching_loss(
                        item["encoded"], item["flow_target_coeff"].detach()
                    )
                else:
                    target = self._baseline_projection(item["direct_baseline"].detach())
                    inv_var = torch.exp(-item["flow_logvar"])
                    flow_loss = flow_loss + ((item["flow_mu"] - target) ** 2 * inv_var + item["flow_logvar"]).mean()
            loss = loss + (flow_loss / len(aux_history)) * self.h.get("lambda_flow", 0.01)

        if aux_history and "bc" in self.loss_fn:
            bc_loss = clean_audio.new_tensor(0.0)
            for item in aux_history:
                if "oracle_alpha" not in item:
                    continue
                alpha_loss = F.mse_loss(item["alpha"], item["oracle_alpha"].detach())
                weight_loss = F.mse_loss(item["weights"], item["oracle_weights"].detach())
                stop_target = item["oracle_stop"].detach()
                reject_target = item["oracle_reject"].detach()
                if "stop_logit" in item and "reject_logit" in item:
                    stop_loss = F.binary_cross_entropy_with_logits(item["stop_logit"], stop_target)
                    reject_loss = F.binary_cross_entropy_with_logits(item["reject_logit"], reject_target)
                else:
                    stop_prob = item["stop_prob"].clamp(1e-6, 1.0 - 1e-6)
                    reject_prob = item["reject_prob"].clamp(1e-6, 1.0 - 1e-6)
                    stop_loss = F.binary_cross_entropy(stop_prob, stop_target)
                    reject_loss = F.binary_cross_entropy(reject_prob, reject_target)
                bc_loss = bc_loss + alpha_loss + weight_loss + 0.2 * stop_loss + 0.2 * reject_loss
            loss = loss + (bc_loss / len(aux_history)) * self.lambda_bc

        if aux_history and "value" in self.loss_fn:
            final_abs = (restored_audio - clean_audio).abs()
            if valid_mask is not None:
                mask = valid_mask.squeeze(1).to(final_abs.device, dtype=final_abs.dtype)
                final_l1 = (final_abs * mask).sum(dim=-1, keepdim=True) / mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
                final_l1 = final_l1.unsqueeze(-1).detach()
            else:
                final_l1 = final_abs.mean(dim=-1, keepdim=True).unsqueeze(-1).detach()
            value_loss = clean_audio.new_tensor(0.0)
            cost_loss = clean_audio.new_tensor(0.0)
            for item in aux_history:
                value_loss = value_loss + F.mse_loss(item["value"], -final_l1)
                cost_loss = cost_loss + F.mse_loss(item["cost_value"], item["morph_cost"].detach())
            loss = loss + ((value_loss + cost_loss) / len(aux_history)) * self.lambda_policy_value

        if aux_history and "risk" in self.loss_fn:
            reject_cost = torch.stack([item["reject_prob"].mean() for item in aux_history]).mean() * self.lambda_reject
            loss = loss + len(aux_history) * self.lambda_step + reject_cost
        return loss

    def forward(self, clean_audio, noisy_audio, valid_mask=None):
        norm_factor = self._norm_factor(noisy_audio)
        clean_audio = (clean_audio * norm_factor).squeeze(1)
        noisy_audio = noisy_audio * norm_factor
        if valid_mask is not None:
            valid_mask = valid_mask.to(noisy_audio.device)

        if self.pcscfm_enabled:
            audio_g, com_g, history = self._pcscfm_restoration(
                noisy_audio,
                clean_audio=clean_audio,
                valid_mask=valid_mask,
                training=True,
            )
            com_g = None
        else:
            audio_g, com_g = self.restore_one_shot(noisy_audio, return_com=True)
            audio_g = audio_g.squeeze(1)
            history = None
        return self._ecg_loss(
            clean_audio,
            audio_g,
            norm_factor,
            predicted_com=com_g,
            aux_history=history,
            valid_mask=valid_mask,
        )


class ECGDenoisingModel(nn.Module):
    block_cls = TSMambaBlock

    def __init__(self, **kwargs):
        super().__init__()
        self.core = MECGECore({"model": kwargs}, block_cls=self.block_cls)

    def forward(self, x):
        return self.core.restore(x)

    @torch.no_grad()
    def denoising(self, x):
        return self.forward(x)

    @torch.no_grad()
    def denoising_with_metadata(self, x, valid_mask=None):
        return self.core.restore_with_metadata(x, valid_mask=valid_mask)

    def compute_loss(self, batch, device, **kwargs):
        noisy, clean = batch[0].to(device), batch[1].to(device)
        valid_mask = batch[2].to(device) if len(batch) > 2 else None
        if noisy.ndim == 2:
            noisy = noisy.unsqueeze(1)
        if clean.ndim == 2:
            clean = clean.unsqueeze(1)
        return self.core(clean, noisy, valid_mask=valid_mask)


@register_model("mecg_e")
class MECGEDenoiser(ECGDenoisingModel):
    block_cls = TSMambaBlock
