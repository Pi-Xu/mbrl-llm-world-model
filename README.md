# TODO

Start with the following command:

```BASH
python main.py exp=llm_dreamer_v3
```

- https://github.com/Eclectic-Sheep/sheeprl/blob/main/howto/register_external_algorithm.md
- https://github.com/Eclectic-Sheep/sheeprl/tree/main/howto

---

### World Model Configuration
The configuration settings for the world model are specified in the file located at `configs/algo/llm_dreamer_S.yaml`. Below is a detailed breakdown of the components within this configuration:

```yaml
world_model:
  recurrent_model:
    recurrent_state_size: 768  # Size of the state in the recurrent model
    dense_units: 768           # Number of units in the dense layer of the recurrent model
  transition_model:
    hidden_size: 768           # Hidden layer size in the transition model
  representation_model:
    hidden_size: 768           # Hidden layer size in the representation model
  hf_model:
    seq_len: 4                 # Sequence length for the Hugging Face model
    peft: true                 # Enable or disable LoRA (Low-Rank Adaptation)
    from_pretrained: false     # Use pretrained model weights (true) or initialize from scratch (false)
```

### Training Configuration

The configuration for training forward steps is defined in `configs/llm_dreamer_v3.yaml`. These settings include the parameters for batch processing and sequence length, which you might need to adjust based on your CUDA memory capabilities:

```yaml
# Algorithm Configuration
algo:
  replay_ratio: 1                        # Ratio of replaying data during training
  total_steps: 5000000                   # Total number of training steps
  per_rank_batch_size: 4                 # Batch size per rank, adjust based on CUDA memory (B shown below)
  per_rank_sequence_length: 4            # Sequence length per rank, adjust based on CUDA memory (L shown below)
  mlp_keys:
    encoder: [state]                     # Encoder keys
    decoder: [state]                     # Decoder keys
```

### Notation:
- **B**: Refers to the batch size, which corresponds to the number of data sequences used in a single training step.
- **L**: Refacts to the sequence length, which is the length of each data sequence.
- **$\{(a_{t}, x_{t}, r_{t})\}^{k+L}_{t=k} \sim D$**: Represents the sampled data sequences used during training, where each sequence includes actions ($a_t$), observations ($x_t$), and rewards ($r_t$) from time step $k$ to $k+L$.
