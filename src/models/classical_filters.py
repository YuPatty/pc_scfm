import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import signal

try:
    from .factory import register_model
except ImportError:
    from factory import register_model


class _ClassicalECGFilter(nn.Module):
    requires_training = False

    def __init__(
        self,
        sampling_rate=250,
        cutoff_hz=0.5,
        filter_kind="fir",
        fir_order=56,
        fir_window="kaiser",
        kaiser_beta=8.6,
        iir_order=1,
        iir_method="butterworth",
        zero_phase=True,
        **kwargs,
    ):
        super().__init__()
        self.sampling_rate = float(sampling_rate)
        self.cutoff_hz = float(cutoff_hz)
        self.filter_kind = str(filter_kind)
        self.fir_order = int(fir_order)
        self.fir_window = str(fir_window)
        self.kaiser_beta = float(kaiser_beta)
        self.iir_order = int(iir_order)
        self.iir_method = str(iir_method).lower()
        self.zero_phase = bool(zero_phase)
        self._dummy = nn.Parameter(torch.zeros(()))

        if self.filter_kind == "fir":
            numtaps = self.fir_order + 1
            window = (self.fir_window, self.kaiser_beta) if self.fir_window == "kaiser" else self.fir_window
            self.b = signal.firwin(
                numtaps,
                self.cutoff_hz,
                pass_zero=False,
                fs=self.sampling_rate,
                window=window,
            ).astype(np.float32)
            self.a = np.array([1.0], dtype=np.float32)
            self.sos = None
        elif self.filter_kind == "iir":
            self.b, self.a, self.sos = self._design_iir()
        else:
            raise ValueError(f"Unsupported filter_kind={filter_kind!r}.")

    def _design_iir(self):
        kwargs = {
            "N": self.iir_order,
            "Wn": self.cutoff_hz,
            "btype": "highpass",
            "fs": self.sampling_rate,
            "output": "sos",
        }
        if self.iir_method == "butterworth":
            sos = signal.butter(**kwargs)
        elif self.iir_method == "chebyshev1":
            sos = signal.cheby1(rp=0.5, **kwargs)
        elif self.iir_method == "chebyshev2":
            sos = signal.cheby2(rs=40.0, **kwargs)
        elif self.iir_method == "elliptic":
            sos = signal.ellip(rp=0.5, rs=40.0, **kwargs)
        else:
            raise ValueError(f"Unsupported iir_method={self.iir_method!r}.")
        b, a = signal.sos2tf(sos)
        return b.astype(np.float32), a.astype(np.float32), sos.astype(np.float32)

    def _filter_1d(self, x):
        padlen = 3 * (max(len(self.a), len(self.b)) - 1)
        if self.sos is not None:
            if self.zero_phase and x.shape[-1] > padlen:
                return signal.sosfiltfilt(self.sos, x, axis=-1).astype(np.float32)
            return signal.sosfilt(self.sos, x, axis=-1).astype(np.float32)

        if self.zero_phase and x.shape[-1] > padlen:
            return signal.filtfilt(self.b, self.a, x, axis=-1).astype(np.float32)
        return signal.lfilter(self.b, self.a, x, axis=-1).astype(np.float32)

    def forward(self, x):
        squeeze = False
        if x.ndim == 2:
            x = x.unsqueeze(1)
            squeeze = True
        if x.ndim != 3 or x.shape[1] != 1:
            raise ValueError(f"Expected ECG shaped [B, T] or [B, 1, T], got {tuple(x.shape)}.")

        device = x.device
        dtype = x.dtype
        filtered = self._filter_1d(x.detach().cpu().numpy())
        output = torch.from_numpy(filtered).to(device=device, dtype=dtype)
        return output.squeeze(1) if squeeze else output

    @torch.no_grad()
    def denoising(self, x):
        return self.forward(x)

    def compute_loss(self, batch, device, **kwargs):
        noisy, clean = batch[0].to(device), batch[1].to(device)
        pred = self.forward(noisy)
        loss = F.mse_loss(pred, clean)
        return loss.detach() + self._dummy * 0.0


@register_model("fir_filter")
class FIRFilterDenoiser(_ClassicalECGFilter):
    def __init__(self, **kwargs):
        kwargs.setdefault("filter_kind", "fir")
        super().__init__(**kwargs)


@register_model("iir_filter")
class IIRFilterDenoiser(_ClassicalECGFilter):
    def __init__(self, **kwargs):
        kwargs.setdefault("filter_kind", "iir")
        super().__init__(**kwargs)
