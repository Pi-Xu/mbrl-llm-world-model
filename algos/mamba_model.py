import huggingface_hub
from transformers import MambaConfig, MambaModel

# TODO: use Pretrain_model
Mamba_config = {
  "_name_or_path": "state-spaces/mamba-130m-hf",
  "architectures": [
    "MambaForCausalLM"
  ],
  "bos_token_id": 0,
  "conv_kernel": 4,
  "d_inner": 1536,
  "d_model": 768,
  "eos_token_id": 0,
  "expand": 2,
  "fused_add_norm": True,
  "hidden_act": "silu",
  "hidden_size": 768,
  "initializer_range": 0.1,
  "intermediate_size": 1536,
  "layer_norm_epsilon": 1e-05,
  "model_type": "mamba",
  "n_layer": 24,
  "num_hidden_layers": 24,
  "pad_token_id": 0,
  "pad_vocab_size_multiple": 8,
  "rescale_prenorm_residual": False,
  "residual_in_fp32": True,
  "rms_norm": True,
  "ssm_cfg": {},
  "state_size": 16,
  "time_step_floor": 0.0001,
  "time_step_init_scheme": "random",
  "time_step_max": 0.1,
  "time_step_min": 0.001,
  "time_step_rank": 48,
  "time_step_scale": 1.0,
  "torch_dtype": "float32",
  "use_bias": False,
  "use_cache": True,
  "use_conv_bias": True,
  # "vocab_size": None
}

def get_mamba_model():
    mamba_config = MambaConfig(**Mamba_config)

    # Forward 时 使用 input embeds
    hf_model = MambaModel(mamba_config)
    return hf_model
