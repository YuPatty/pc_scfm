import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .factory import register_model
except ImportError:
    from factory import register_model


def _match_length(x, target_length):
    current_length = x.shape[-1]
    if current_length == target_length:
        return x
    if current_length > target_length:
        return x[..., :target_length]
    return F.pad(x, (0, target_length - current_length))


class _FCNDAEConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, activate=True):
        super().__init__()
        self.stride = int(stride)
        self.kernel_size = int(kernel_size)
        self.activate = bool(activate)
        self.conv = nn.Conv1d(
            int(in_channels),
            int(out_channels),
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=(self.kernel_size - self.stride) // 2,
        )
        self.norm = nn.BatchNorm1d(int(out_channels)) if self.activate else nn.Identity()
        self.activation = nn.ELU() if self.activate else nn.Identity()

    def forward(self, x):
        target_length = x.shape[-1] if self.stride == 1 else None
        x = self.conv(x)
        if target_length is not None:
            x = _match_length(x, target_length)
        return self.activation(self.norm(x))


class _FCNDAEDeconvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, activate=True):
        super().__init__()
        self.activate = bool(activate)
        self.deconv = nn.ConvTranspose1d(
            int(in_channels),
            int(out_channels),
            kernel_size=int(kernel_size),
            stride=int(stride),
            padding=(int(kernel_size) - int(stride)) // 2,
        )
        self.norm = nn.BatchNorm1d(int(out_channels)) if self.activate else nn.Identity()
        self.activation = nn.ELU() if self.activate else nn.Identity()

    def forward(self, x, target_length):
        x = self.deconv(x)
        x = _match_length(x, target_length)
        return self.activation(self.norm(x))


@register_model("fcn_dae")
class FCNDAEDenoiser(nn.Module):
    """FCN-based denoising autoencoder from Chiang et al., IEEE Access 2019."""

    def __init__(
        self,
        input_channels=1,
        kernel_size=16,
        encoder_channels=(40, 20, 20, 20, 40, 1),
        encoder_strides=(2, 2, 2, 2, 2, 1),
        decoder_channels=(1, 40, 20, 20, 20, 40, 1),
        decoder_strides=(1, 2, 2, 2, 2, 2, 1),
        loss_fn="mse",
        **kwargs,
    ):
        super().__init__()
        if loss_fn != "mse":
            raise ValueError("FCNDAEDenoiser currently supports only model.loss_fn='mse'.")
        if len(encoder_channels) != len(encoder_strides):
            raise ValueError("encoder_channels and encoder_strides must have the same length.")
        if len(decoder_channels) != len(decoder_strides):
            raise ValueError("decoder_channels and decoder_strides must have the same length.")

        self.encoder_strides = tuple(int(stride) for stride in encoder_strides)
        self.decoder_strides = tuple(int(stride) for stride in decoder_strides)
        self.downsample_count = sum(stride == 2 for stride in self.encoder_strides)

        encoder = []
        in_channels = int(input_channels)
        for out_channels, stride in zip(encoder_channels, self.encoder_strides):
            encoder.append(_FCNDAEConvBlock(in_channels, out_channels, kernel_size, stride))
            in_channels = int(out_channels)
        self.encoder = nn.ModuleList(encoder)

        decoder = []
        in_channels = int(encoder_channels[-1])
        last_index = len(decoder_channels) - 1
        for index, (out_channels, stride) in enumerate(zip(decoder_channels, self.decoder_strides)):
            decoder.append(
                _FCNDAEDeconvBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    activate=index != last_index,
                )
            )
            in_channels = int(out_channels)
        self.decoder = nn.ModuleList(decoder)

    def forward(self, x):
        squeeze = False
        if x.ndim == 2:
            x = x.unsqueeze(1)
            squeeze = True
        if x.ndim != 3 or x.shape[1] != 1:
            raise ValueError(f"Expected ECG shaped [B, T] or [B, 1, T], got {tuple(x.shape)}.")

        original_length = x.shape[-1]
        encoder_lengths = []
        z = x
        for layer in self.encoder:
            z = layer(z)
            encoder_lengths.append(z.shape[-1])

        target_lengths = [encoder_lengths[-1]]
        target_lengths.extend(reversed(encoder_lengths[:-1]))
        target_lengths.append(original_length)

        y = z
        for layer, target_length in zip(self.decoder, target_lengths):
            y = layer(y, target_length)
        return y.squeeze(1) if squeeze else y

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

        pred = self.forward(noisy)
        loss = F.mse_loss(pred, clean, reduction="none")
        if valid_mask is not None:
            valid_mask = valid_mask.to(device=device, dtype=loss.dtype)
            loss = (loss * valid_mask).sum() / valid_mask.sum().clamp_min(1.0)
        else:
            loss = loss.mean()
        return loss
