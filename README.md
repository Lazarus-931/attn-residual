

# Attention Residuals

This is a small-scale replication of [Attention Residual](https://arxiv.org/abs/2603.15031) on a single v6 TPU. This will create, run and benchmark standard Kimi Linear against Kimi Linear with Attention Residuals(Full/Block) with identical configurations.

Specifications:
- 300M Param Model
- 256-512 Batch Size
- 1024-2048 Context Length

(Soon to change)


There will be a Kimi Linear baseline and a (+ AttnRes) variant. They will be trained individually across a single v6 TPU using jax, looking to verify and see the supposed benefit of AttnResiduals from the paper.


Pre-training occurs using 4090-context window, Muon[1] and a WSG(Warmup-Stable-Decay) learning rate schedule with a gloabl batch size of 8M tokens.

> *"Our architecture is identical to Kimi Linear ...., which interleaves Kimi Delta Attention (KDA) and Multi-Head Latent Attention (MLA) layers in a 3:1 ratio, each followed by an MoE feed-forward layer. The only modification is the addition of AttnRes to the residual connections; all other components ... remain unchanged."* - *__Authors__*

[1]: https://kellerjordan.github.io/posts/muon/



*Question*: **Are Attention Residuals findings consistent across single tru training? Furthermore, can a 300M parameter MoE Tranfsormer with Attention Residuals bring lower validation loss compared to PreNorm? And as an extension, how can it best be optimized for single device training?**

Some questions I also explore in this small-scale experimentation sprint:
1. Do deeper layers attend to earlier layers?
2. Does the distribution flatten or sharpen during training?
3. Is AttenRes using many layers or colapsing into one instead(high vs low entropy)?

Further things to possibly explore: Muon vs AdamW vs Prodigy | WSD vs Cosine | What if Attention Residual per Top and Bottom Block instead of between each attn and moe layer?


## Note

I actually switched to H100 since I found out I able to cut my training time x2




## Notes along the way

### Early Findings - Baseline testing
So I tried a small baseline run:


``` python
class ModelArg:

    dim: int = 1024
    inter_dim: int = 1536
    heads: int = 8
    local_heads: int = 8
    head_dim: int = 128
    vocab_size: int = 32768

    # MLA
    q_lora_rank: int = 256
    kv_lora_rank: int = 64
    qk_head_dim: int = 128
    qk_nope_head_dim: int = 64
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128

    # MoE
    n_routed_experts: int = 4
    n_activated_experts: int = 2
    n_groups: int = 1
    topk_groups: int = 1
    shared_experts: int = 1
    score_func: Literal["softmax", "sigmoid"] = "sigmoid"
    scaling_factor: float = 2.446

    # Architecture
    n_layer_groups: int = 1
    block_size: int = 128
    max_seq: int = 1024

    # Data
    dataset_name: str = "HuggingFaceFW/fineweb-edu"
    dataset_subset: str = "sample-10BT"
    dataset_split: str = "train"
    streaming: bool = True
    text_field: str = "text"
    seed: int = 0
    shuffle_buffer_size: int = 10_000
    max_examples: int | None = None

    # Training
    model_type: Literal["baseline", "full", "batch"] = "baseline"
    tokenizer_name: str = "gpt2"
    use_wandb: bool = True
    wandb_project: str = "attn-residual"
    wandb_entity: str | None = None
    wandb_run_name: str | None = None
    log_every_steps: int = 10
    residual_log_every_steps: int = 50

    lr: float = 3e-4
    min_lr: float = 1e-5
    batch_size: int = 4
    

    total_steps: int = 10_000


    warmup_steps: int = 300
    hold_steps: int = 700

    # Muon
    muon_beta: float = 0.95
    muon_ns_steps: int = 5
    muon_lr: float = 0.01

    # Adam
    adam_b1: float = 0.9
    adam_b2: float = 0.95
    adam_eps: float = 1e-8
    weight_decay: float = 0.01

    # Grad clipping
    max_grad_norm: float = 1.0
```

and the result was a consistent learning, then a complete lack of model learning, then a straight exploding gradient, from my understanding:

<img width="1357" height="886" alt="Screenshot 2026-03-25 at 12 59 04 PM" src="https://github.com/user-attachments/assets/3c95a02d-8580-4f38-bbb0-957e878ff558" />


I already gave the Muon a high lr, 0.01, and kept 1e-4 for adamw, I theorize, it might be a deeper numerical problem in the KDA layer. THis is because I've tried endless tweakes in the lr, warmup, decay but they always
appear to descend to NaN. 


