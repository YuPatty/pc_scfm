import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .factory import register_model
except ImportError:
    from factory import register_model


class _ConvBranch(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, activation="linear"):
        super().__init__()
        padding = int(dilation) * (int(kernel_size) - 1) // 2
        self.conv = nn.Conv1d(
            int(in_channels),
            int(out_channels),
            kernel_size=int(kernel_size),
            stride=1,
            padding=padding,
            dilation=int(dilation),
        )
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        if self.activation == "relu":
            return F.relu(x)
        if self.activation == "linear":
            return x
        raise ValueError(f"Unsupported activation: {self.activation}")


class LANLFilterModule(nn.Module):
    def __init__(self, in_channels, layers, kernels=(3, 5, 9, 15)):
        super().__init__()
        branch_channels = int(layers) // (2 * len(kernels))
        self.out_channels = branch_channels * 2 * len(kernels)
        self.branches = nn.ModuleList(
            [
                _ConvBranch(in_channels, branch_channels, kernel, activation="linear")
                for kernel in kernels
            ]
            + [
                _ConvBranch(in_channels, branch_channels, kernel, activation="relu")
                for kernel in kernels
            ]
        )

    def forward(self, x):
        return torch.cat([branch(x) for branch in self.branches], dim=1)


class DilatedLANLFilterModule(nn.Module):
    def __init__(self, in_channels, layers, kernels=(5, 9, 15), dilation=3):
        super().__init__()
        branch_channels = int(layers) // (2 * len(kernels))
        self.out_channels = branch_channels * 2 * len(kernels)
        self.branches = nn.ModuleList(
            [
                _ConvBranch(in_channels, branch_channels, kernel, dilation=dilation, activation="linear")
                for kernel in kernels
            ]
            + [
                _ConvBranch(in_channels, branch_channels, kernel, dilation=dilation, activation="relu")
                for kernel in kernels
            ]
        )

    def forward(self, x):
        return torch.cat([branch(x) for branch in self.branches], dim=1)


class _DeepFilterBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        layers,
        dilated=False,
        kernels=(3, 5, 9, 15),
        dilated_kernels=(5, 9, 15),
        dilation=3,
        dropout=0.4,
    ):
        super().__init__()
        if dilated:
            self.filter = DilatedLANLFilterModule(
                in_channels,
                layers,
                kernels=dilated_kernels,
                dilation=dilation,
            )
        else:
            self.filter = LANLFilterModule(in_channels, layers, kernels=kernels)
        self.dropout = nn.Dropout(float(dropout))
        self.norm = nn.BatchNorm1d(self.filter.out_channels)
        self.out_channels = self.filter.out_channels

    def forward(self, x):
        return self.norm(self.dropout(self.filter(x)))


@register_model("deepfilter")
class DeepFilterDenoiser(nn.Module):
    """PyTorch port of the official DeepFilter multibranch LANL-dilated model."""

    def __init__(
        self,
        input_channels=1,
        layers=(64, 64, 32, 32, 16, 16),
        dilated_pattern=(False, True, False, True, False, True),
        kernels=(3, 5, 9, 15),
        dilated_kernels=(5, 9, 15),
        dilation=3,
        dropout=0.4,
        output_kernel_size=9,
        loss_fn="mse",
        **kwargs,
    ):
        super().__init__()
        if loss_fn != "mse":
            raise ValueError("DeepFilterDenoiser currently supports only model.loss_fn='mse'.")
        if len(layers) != len(dilated_pattern):
            raise ValueError("layers and dilated_pattern must have the same length.")

        blocks = []
        in_channels = int(input_channels)
        for layer_width, is_dilated in zip(layers, dilated_pattern):
            block = _DeepFilterBlock(
                in_channels,
                layer_width,
                dilated=bool(is_dilated),
                kernels=kernels,
                dilated_kernels=dilated_kernels,
                dilation=dilation,
                dropout=dropout,
            )
            blocks.append(block)
            in_channels = block.out_channels
        self.blocks = nn.Sequential(*blocks)
        self.output = nn.Conv1d(
            in_channels,
            1,
            kernel_size=int(output_kernel_size),
            stride=1,
            padding=(int(output_kernel_size) - 1) // 2,
        )

    def forward(self, x):
        squeeze = False
        if x.ndim == 2:
            x = x.unsqueeze(1)
            squeeze = True
        if x.ndim != 3 or x.shape[1] != 1:
            raise ValueError(f"Expected ECG shaped [B, T] or [B, 1, T], got {tuple(x.shape)}.")

        y = self.output(self.blocks(x))
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
            if valid_mask.ndim == 2:
                valid_mask = valid_mask.unsqueeze(1)
            loss = (loss * valid_mask).sum() / valid_mask.sum().clamp_min(1.0)
        else:
            loss = loss.mean()
        return loss
