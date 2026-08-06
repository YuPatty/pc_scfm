import torch.nn as nn

try:
    from .factory import register_model
    from .mecg_e import ECGDenoisingModel, TSMambaBlock
except ImportError:
    from factory import register_model
    from mecg_e import ECGDenoisingModel, TSMambaBlock


class AttentionModule(nn.Module):
    def __init__(self, dim, n_head=8, dropout=0.0):
        super().__init__()
        self.layernorm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_head, dropout=dropout, batch_first=True)

    def forward(self, x):
        x_norm = self.layernorm(x)
        out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        return out


class MambAttentionBlock(TSMambaBlock):
    def __init__(self, h):
        super().__init__(h)
        self.use_time_attention = h.get("use_time_attention", True)
        self.use_freq_attention = h.get("use_freq_attention", True)
        self.attention_position = h.get("attention_position", "before_mamba")
        if self.attention_position not in {"before_mamba", "after_mamba"}:
            raise ValueError(
                "attention_position must be either 'before_mamba' or 'after_mamba'."
            )
        self.attention = AttentionModule(
            dim=h.dense_channel,
            n_head=h.get("attention_heads", 8),
            dropout=h.get("attention_dropout", 0.0),
        )

    def forward(self, x):
        b, c, t, f = x.size()
        x = x.permute(0, 3, 2, 1).contiguous().view(b * f, t, c)
        if self.use_time_attention and self.attention_position == "before_mamba":
            x = self.attention(x) + x
        x = self.tlinear(self.time_mamba(x).permute(0, 2, 1)).permute(0, 2, 1) + x
        if self.use_time_attention and self.attention_position == "after_mamba":
            x = self.attention(x) + x
        x = x.view(b, f, t, c).permute(0, 2, 1, 3).contiguous().view(b * t, f, c)
        if self.use_freq_attention and self.attention_position == "before_mamba":
            x = self.attention(x) + x
        x = self.flinear(self.freq_mamba(x).permute(0, 2, 1)).permute(0, 2, 1) + x
        if self.use_freq_attention and self.attention_position == "after_mamba":
            x = self.attention(x) + x
        return x.view(b, t, f, c).permute(0, 3, 1, 2)


@register_model("mambattention_ecg")
class MambAttentionECGDenoiser(ECGDenoisingModel):
    block_cls = MambAttentionBlock
