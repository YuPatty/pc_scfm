import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .factory import register_model
except ImportError:
    from factory import register_model


@register_model("drnn")
class DRNNDenoiser(nn.Module):
    def __init__(
        self,
        input_size=1,
        hidden_size=64,
        lstm_layers=1,
        dense_layers=(64, 64),
        dropout=0.0,
        residual=False,
        **kwargs,
    ):
        super().__init__()
        self.residual = bool(residual)
        self.lstm = nn.LSTM(
            input_size=int(input_size),
            hidden_size=int(hidden_size),
            num_layers=int(lstm_layers),
            batch_first=True,
            dropout=float(dropout) if int(lstm_layers) > 1 else 0.0,
        )

        layers = []
        in_features = int(hidden_size)
        for width in dense_layers:
            layers.append(nn.Linear(in_features, int(width)))
            layers.append(nn.ReLU())
            in_features = int(width)
        layers.append(nn.Linear(in_features, 1))
        self.head = nn.Sequential(*layers)

    def forward(self, x):
        squeeze = False
        if x.ndim == 2:
            x = x.unsqueeze(1)
            squeeze = True
        if x.ndim != 3 or x.shape[1] != 1:
            raise ValueError(f"Expected ECG shaped [B, T] or [B, 1, T], got {tuple(x.shape)}.")

        sequence = x.transpose(1, 2)
        features, _ = self.lstm(sequence)
        restored = self.head(features).transpose(1, 2)
        if self.residual:
            restored = x + restored
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

        pred = self.forward(noisy)
        loss = F.mse_loss(pred, clean, reduction="none")
        if valid_mask is not None:
            valid_mask = valid_mask.to(device=device, dtype=loss.dtype)
            loss = (loss * valid_mask).sum() / valid_mask.sum().clamp_min(1.0)
        else:
            loss = loss.mean()
        return loss
