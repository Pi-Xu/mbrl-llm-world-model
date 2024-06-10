import huggingface_hub
from transformers import MambaConfig, MambaModel, AutoModelForCausalLM
from peft import LoraConfig, TaskType, get_peft_model

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



# PEFT_config = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)

PERF_config =  LoraConfig(
    r=8,
    target_modules=["x_proj", "embeddings", "in_proj", "out_proj"],
    task_type=TaskType.CAUSAL_LM,
    bias="none"
)

def get_mamba_model(from_pretrained=False):
    mamba_config = MambaConfig(**Mamba_config)

    # Forward 时 使用 input embeds
    hf_model = MambaModel(mamba_config)
    if from_pretrained:
        hf_model = hf_model.from_pretrained("state-spaces/mamba-130m-hf")
    # PEFT
    hf_model = get_peft_model(hf_model,peft_config=PERF_config)
    hf_model.print_trainable_parameters()
    return hf_model
