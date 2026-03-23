

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
