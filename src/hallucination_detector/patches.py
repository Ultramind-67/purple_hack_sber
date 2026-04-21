"""
Patches for loading GigaChat (DeepseekV3 architecture).
Fixes type casting and q_lora_rank=None issues.
"""
import torch.nn as nn


def apply_patches():
    """Apply all necessary patches for GigaChat model loading."""
    import transformers.models.deepseek_v3.configuration_deepseek_v3 as cfg_mod
    _orig_cfg_init = cfg_mod.DeepseekV3Config.__init__

    def _patched_cfg_init(self, *a, **kw):
        if 'routed_scaling_factor' in kw:
            kw['routed_scaling_factor'] = float(kw['routed_scaling_factor'])
        rs = kw.get('rope_scaling')
        if rs:
            for k in ['beta_fast', 'beta_slow', 'factor']:
                if k in rs and isinstance(rs[k], int):
                    rs[k] = float(rs[k])
        _orig_cfg_init(self, *a, **kw)

    cfg_mod.DeepseekV3Config.__init__ = _patched_cfg_init

    import transformers.models.deepseek_v3.modeling_deepseek_v3 as dsv3
    _orig_attn_init = dsv3.DeepseekV3Attention.__init__

    def _patched_attn_init(self, config, layer_idx):
        saved = config.q_lora_rank
        if saved is None:
            config.q_lora_rank = config.hidden_size
        try:
            _orig_attn_init(self, config, layer_idx)
        finally:
            config.q_lora_rank = saved
        if saved is None:
            del self.q_a_proj
            del self.q_a_layernorm
            del self.q_b_proj
            qhd = config.qk_nope_head_dim + config.qk_rope_head_dim
            self.q_proj = nn.Linear(
                config.hidden_size,
                config.num_attention_heads * qhd,
                bias=config.attention_bias
            )
            self.q_lora_rank = None

    dsv3.DeepseekV3Attention.__init__ = _patched_attn_init

    _orig_attn_fwd = dsv3.DeepseekV3Attention.forward

    def _patched_attn_fwd(self, hidden_states, *a, **kw):
        if getattr(self, 'q_lora_rank', -1) is None:
            object.__setattr__(self, 'q_a_proj', lambda x: x)
            object.__setattr__(self, 'q_a_layernorm', lambda x: x)
            object.__setattr__(self, 'q_b_proj', self.q_proj)
            try:
                return _orig_attn_fwd(self, hidden_states, *a, **kw)
            finally:
                for attr in ['q_a_proj', 'q_a_layernorm', 'q_b_proj']:
                    object.__delattr__(self, attr)
        return _orig_attn_fwd(self, hidden_states, *a, **kw)

    dsv3.DeepseekV3Attention.forward = _patched_attn_fwd
    print("✅ GigaChat patches applied")
