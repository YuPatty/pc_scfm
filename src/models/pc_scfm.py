try:
    from .factory import register_model
    from .mambattention import MambAttentionECGDenoiser
except ImportError:
    from factory import register_model
    from mambattention import MambAttentionECGDenoiser


@register_model("pc_scfm")
class PCSCFMDenoiser(MambAttentionECGDenoiser):
    def __init__(self, **kwargs):
        kwargs["pcscfm_enabled"] = True
        super().__init__(**kwargs)
