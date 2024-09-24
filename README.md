# LLM Dreamer V3 Project

## Getting Started

To get started with the LLM Dreamer V3 project, clone the repository. Before running the main application, install the required dependencies with the following command:

```bash
pip install -r requirements.txt
```

Then, start the project with this command:

```bash
python main.py exp=llm_dreamer_v3
```

For more detailed instructions on how to register external algorithms and set up your environment, refer to the following guides:

- [Registering External Algorithms](https://github.com/Eclectic-Sheep/sheeprl/blob/main/howto/register_external_algorithm.md)
- [How-to Guides](https://github.com/Eclectic-Sheep/sheeprl/tree/main/howto)

## Contribution and Innovation

This project aims to integrate Large Language Models (LLMs) into the reinforcement learning (RL) framework of Dreamer V3. The primary contributions include:


1. **Model Modification**: We have introduced a multimodal approach to the World Model by incorporating a language token (lt) into the input and output layers.
2. **Algorithmic Enhancements**: We have replaced the simple structure of the original world model with a more complex pre-trained model, specifically using Mamba, which fits well into the Partially Observable Markov Decision Process (POMDP) structure.
3. **Challenges**: The original world model structures are relatively simple, but pre-trained models like LLMs are more complex, leading to potential training inefficiencies in online scenarios.
4. **Innovation**: To demonstrate the effectiveness of our approach, we aim to show improved performance in challenging environments, similar to comparisons with Dreamer V3.


## World Model Configuration

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

## Training Configuration

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

## Notation:

- **B**: Refers to the batch size, which corresponds to the number of data sequences used in a single training step.
- **L**: Refers to the sequence length, which is the length of each data sequence.
- **$\{(a_{t}, x_{t}, r_{t})\}^{k+L}_{t=k} \sim D$**: Represents the sampled data sequences used during training, where each sequence includes actions ($a_t$), observations ($x_t$), and rewards ($r_t$) from time step $k$ to $k+L$.

<!-- ## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. -->

## Acknowledgments

We would like to acknowledge the contributions of the following repositories and papers:

- [SheepRL](https://github.com/Eclectic-Sheep/sheeprl) for providing the framework and guides.
- [Unleashing the Power of Pre-trained Language Models for Offline Reinforcement Learning](https://arxiv.org/abs/2404.01234) for pioneering the use of LLMs in RL.
- [Understanding Language in the World by Predicting the Future](https://arxiv.org/abs/2404.05678) for insights into integrating language models with world models.
