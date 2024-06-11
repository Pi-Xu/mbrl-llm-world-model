import huggingface_hub
from transformers import GPT2Config, GPT2Model
from peft import LoraConfig, TaskType, get_peft_model

# PEFT_config = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)

# borrow from https://github.com/srzer/LaMo-2023/blob/main/experiment-atari/model.py#L120
PERF_config =  LoraConfig(
    r=8,
    target_modules=["c_attn"], 
    lora_alpha=32,
    # task_type=TaskType.CAUSAL_LM,
    bias="none"
)

def get_model(from_pretrained=False, peft = False):
    config = GPT2Config(
    vocab_size=50257,  # GPT-2 的词汇表大小
    n_positions=1024,  # 序列的最大长度
    n_ctx=1024,        # 上下文的最大长度
    n_embd=768,        # 嵌入层的维度
    n_layer=12,        # Transformer 的层数
    n_head=12          # 每个 Transformer 层的注意力头的数量
    )

    # Forward 时 使用 input embeds
    hf_model = GPT2Model(config)
    if from_pretrained:
        hf_model = hf_model.from_pretrained("gpt2")
    # PEFT
    if peft:
      hf_model = get_peft_model(hf_model,peft_config=PERF_config)
      hf_model.print_trainable_parameters()
    return hf_model
