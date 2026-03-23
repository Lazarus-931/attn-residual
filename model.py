import jax
import jax.numpy as jnp
import jax.lax as lax
import flax.linen as nn
import optax
import functools
from typing import Optional, Tuple, Literal, NamedTuple, Any
from dataclasses import dataclass
from einops import rearrange


@dataclass
class ModelArg:
  """
  Model Arguments for Kimi Linear

  Configurations for training a MoE Transformer, aligned closely
  with [Deepseek et al](https://arxiv.org/abs/2502.16982). Configured
  to run as efficient as possible on a single v6 TPU.
  """
  # Model dimensions (~310M total, ~120M active/token)
  dim: int = 1024
  inter_dim: int = 1536
  heads: int = 8
  local_heads: int = 8
  head_dim: int = 128
  vocab_size: int = 32768

  # MLA (Multi-head Latent Attention)
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
  score_func: Literal['softmax', 'sigmoid'] = 'sigmoid'
  scaling_factor: float = 2.446

  # Architecture
  n_layer_groups: int = 3
  block_size: int = 128
  max_seq: int = 4096

  # Training
  lr: float = 2.5e-3
  batch_size: int = 512
  weight_decay: float = 0.01
  warmup_steps: int = 1000


# Utils

def exists(x):
  return x is not None

def default(val, d):
  if exists(val):
    return val
  return d() if callable(d) else d

# RMSNorm

class RMSNorm(nn.Module):
  """
  Root Mean Squared Normalization
  """
  dim: int
  eps: float = 1e-6

  def setup(self):
      self.weight = self.param('weight', nn.initializers.ones, (self.dim,))

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
      # Norm in f32 for stability, output in input dtype
      orig_dtype = x.dtype
      x = x.astype(jnp.float32)
      rms = jnp.sqrt(jnp.mean(x ** 2, axis=-1, keepdims=True) + self.eps)
      return ((x / rms) * self.weight).astype(orig_dtype)

# Linear(s)

def linear(x: jnp.ndarray, weights: jnp.ndarray, bias: Optional[jnp.ndarray] = None) -> jnp.ndarray:
  out = jnp.dot(x, weights)
  if exists(bias):
    out = out + bias
  return out


class Linear(nn.Module):
  in_features: int
  out_features: int
  use_bias: bool = False

  @nn.compact
  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    weights = self.param(
        'weights',
        nn.initializers.lecun_normal(),
        (self.in_features, self.out_features)
    )
    bias = None
    if self.use_bias:
      bias = self.param(
          'bias',
          nn.initializers.zeros,
          (self.out_features,)
      )
    return linear(x, weights, bias)



# Short Convolution

class ShortConv(nn.Module):
  """
  Short Convolution function
  """

  @nn.compact
  def __call__(self, x: jnp.ndarray, weights: jnp.ndarray, bias: jnp.ndarray, k: int = 4) -> jnp.ndarray:
    """
    Forward call for a short convolution
    """

    batch, seq, dim = x.shape

    x_padded = jnp.pad(x, ((0,0), (k-1, 0), (0,0)))

    out = jnp.zeros((batch, seq, dim))
    for i in range(k):
        out = out + x_padded[:, i:i+seq, :] * weights[None, None, :, i].squeeze(-1)

    return out + bias




# Chunked Kimi Delta Attention

class KDA(nn.Module):
  """
  Chunked Kimi Delta Attention
  """
  dim: int
  heads: int
  head_dim: int = 128
  chunk_size: int = 64

  @nn.compact
  def __call__(self,
               q: jnp.ndarray,
               k: jnp.ndarray,
               v: jnp.ndarray,
               g: jnp.ndarray,
               beta: jnp.ndarray,
               initial_state: Optional[jnp.ndarray] = None,
               ) -> tuple[jnp.ndarray, jnp.ndarray]:

      B, T, H, K = q.shape
      V = v.shape[-1]
      C = self.chunk_size
      N = T // C
      assert T % C == 0


      q, k, v, g, beta = [x.astype(jnp.bfloat16) for x in (q, k, v, g, beta)]


      q, k, v, g = [rearrange(x, 'b (n c) h d -> b h n c d', c=C)
                    for x in (q, k, v, g)]
      beta = rearrange(beta, 'b (n c) h -> b h n c', c=C)

      q = q * (K ** -0.5)
      g = jnp.cumsum(g, axis=-2)


      A = jnp.einsum('...jd,...id->...ji',
                      k * jnp.exp(g),
                      k * jnp.exp(-g),
                      precision=jax.lax.Precision.HIGHEST)

      A = A * beta[..., None]


      mask_diag = jnp.tril(jnp.ones((C, C), dtype=jnp.bool_))
      A = -jnp.where(mask_diag, A, 0.0)


      def fwd_sub_step(i, A):
          col    = A[..., i:i+1, :]
          update = (col * A).sum(-2)
          return A.at[..., i, :].add(update)

      A = jax.lax.fori_loop(1, C, fwd_sub_step, A)


      A = (A + jnp.eye(C, dtype=jnp.bfloat16)) * beta[..., None, :]


      w = jnp.einsum('...ij,...jd->...id', A, jnp.exp(g) * k,
                      precision=jax.lax.Precision.HIGHEST)
      u = jnp.einsum('...ij,...jd->...id', A, v,
                      precision=jax.lax.Precision.HIGHEST)

      S_init = jnp.zeros((B, H, K, V), dtype=jnp.bfloat16)
      if initial_state is not None:
          S_init = S_init + initial_state.astype(jnp.bfloat16)

      mask_strict = jnp.triu(jnp.ones((C, C), dtype=jnp.bool_), k=1)

      def chunk_step(S, inputs):
          q_i, k_i, u_i, g_i, w_i = inputs


          A_qk = jnp.einsum('...id,...jd->...ij',
                             q_i * jnp.exp(g_i),
                             k_i * jnp.exp(-g_i),
                             precision=jax.lax.Precision.HIGHEST)
          A_qk = jnp.where(mask_strict, 0.0, A_qk)


          v_i = u_i - jnp.einsum('...id,...dv->...iv', w_i, S,
                                  precision=jax.lax.Precision.HIGHEST)


          o_i = (jnp.einsum('...id,...dv->...iv',
                             q_i * jnp.exp(g_i), S,
                             precision=jax.lax.Precision.HIGHEST)
               + jnp.einsum('...ij,...jv->...iv',
                             A_qk, v_i,
                             precision=jax.lax.Precision.HIGHEST))


          decay = g_i[..., -1:, :]
          S_new = (S * jnp.exp(rearrange(decay, 'b h 1 k -> b h k 1'))
                 + jnp.einsum('...id,...iv->...dv',
                              k_i * jnp.exp(-g_i), v_i,
                              precision=jax.lax.Precision.HIGHEST))

          return S_new, o_i


      S_final, o = jax.lax.scan(
          chunk_step,
          S_init,
          (q, k, u, g, w)
      )

      return (
          rearrange(o, 'b h n c v -> b (n c) h v').astype(jnp.bfloat16),
          S_final
      )


# Muon - Optimizer

def newton_schulz(g: jnp.ndarray, steps: int = 5, eps: float = 1e-7):
  """
  Newton-Schultz iteration on 2D matrix
  """
  assert g.ndim >= 2

  a, b, c = 3.4445, -4.7750, 2.0315

  x = g.astype(jnp.float32)

  x = x / (jnp.linalg.norm(x, axis=(-2, -1), keepdims=True) + eps)

  def _ns_step(_, x):
      A = x @ x.T
      B = b * A + c * A @ A
      return a * x + B @ x

  x = jax.lax.fori_loop(0, steps, _ns_step, x)

  if g.shape[-2] > g.shape[-1]:
      x = x.T

  return x

class MuonState(NamedTuple):
    momentum_buffer: Any
    count: jnp.ndarray


def init_muon(params: Any) -> MuonState:
    return MuonState(
        momentum_buffer=jax.tree_util.tree_map(
            lambda p: jnp.zeros_like(p, dtype=jnp.bfloat16), params
        ),
        count=jnp.zeros([], jnp.int32),
    )


def muon_step(
    grads: Any,
    state: MuonState,
    learning_rate: float = 0.02,
    beta: float = 0.95,
    nesterov: bool = True,
    ns_steps: int = 5,
) -> tuple[Any, MuonState]:
    results = jax.tree_util.tree_map(
        lambda g, b: muon_update(g, b, beta=beta, ns_steps=ns_steps, nesterov=nesterov),
        grads, state.momentum_buffer,
    )
    updates = jax.tree_util.tree_map(lambda r: learning_rate * r[0], results)
    new_buf = jax.tree_util.tree_map(lambda r: r[1], results)

    return updates, MuonState(momentum_buffer=new_buf, count=state.count + 1)



def muon_update(
    grad: jnp.ndarray,
    buf: jnp.ndarray,
    beta: float = 0.95,
    ns_steps: int = 5,
    nesterov: bool = True,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    g = grad.astype(jnp.float32)
    b = buf.astype(jnp.float32)

    new_buf = beta * b + (1.0 - beta) * g
    update = beta * new_buf + (1.0 - beta) * g if nesterov else new_buf

    if grad.ndim == 2:
        update = newton_schulz(update, steps=ns_steps)
        m, n = update.shape
        update = update * max(1, m / n) ** 0.5
    elif grad.ndim >= 3:
        update = jax.vmap(functools.partial(newton_schulz, steps=ns_steps))(update)
        m, n = update.shape[-2], update.shape[-1]
        update = update * max(1, m / n) ** 0.5

    return update.astype(grad.dtype), new_buf.astype(buf.dtype)



# Multi Linear Attention

class multihead_attn(nn.Module):
  """
  Multi-head Latent Attention (naive variant, no KV cache)

  Uses LoRA-compressed Q projection and joint KV compression.
  Q = [q_nope, q_pe], K = [k_nope, k_pe], split into rope/non-rope parts.
  """
  dim: int
  heads: int
  local_heads: int
  q_lora_rank: int
  kv_lora_rank: int
  qk_head_dim: int = 128
  v_head_dim: int = 128
  qk_nope_head_dim: int = 64
  qk_rope_head_dim: int = 64
  max_seq: int = 4096

  def setup(self):
    self.softmax_scale = self.qk_head_dim ** -0.5

    if self.q_lora_rank == 0:
      self.wq = Linear(in_features=self.dim, out_features=self.heads * self.qk_head_dim)
    else:
      self.wq_a = Linear(in_features=self.dim, out_features=self.q_lora_rank)
      self.q_norm = RMSNorm(self.q_lora_rank)
      self.wq_b = Linear(in_features=self.q_lora_rank, out_features=self.heads * self.qk_head_dim)

    self.wkv_a = Linear(in_features=self.dim, out_features=self.kv_lora_rank + self.qk_rope_head_dim)
    self.kv_norm = RMSNorm(self.kv_lora_rank)
    self.wkv_b = Linear(in_features=self.kv_lora_rank,
                        out_features=self.heads * (self.qk_nope_head_dim + self.v_head_dim))
    self.wo = Linear(in_features=self.heads * self.v_head_dim, out_features=self.dim)

  def __call__(self, x: jnp.ndarray, start: int = 0,
               freqs_cis: Optional[jnp.ndarray] = None,
               mask: Optional[jnp.ndarray] = None):
    batch, seqlen, _ = x.shape

    # Q projection (direct or LoRA-compressed)
    if self.q_lora_rank == 0:
      q = self.wq(x)
    else:
      q = self.wq_b(self.q_norm(self.wq_a(x)))

    q = q.reshape(batch, seqlen, self.local_heads, self.qk_head_dim)
    q_nope, q_pe = jnp.split(q, [self.qk_nope_head_dim], axis=-1)

    # KV joint compression
    kv = self.wkv_a(x)
    kv, k_pe = jnp.split(kv, [self.kv_lora_rank], axis=-1)

    # Naive MLA: no absorbed KV, just expand
    q = jnp.concatenate([q_nope, q_pe], axis=-1)

    kv = self.wkv_b(self.kv_norm(kv))
    kv = kv.reshape(batch, seqlen, self.local_heads, self.qk_nope_head_dim + self.v_head_dim)

    k_nope, v = jnp.split(kv, [self.qk_nope_head_dim], axis=-1)
    k = jnp.concatenate([k_nope, jnp.broadcast_to(
        k_pe[:, :, None, :],
        (batch, seqlen, self.local_heads, self.qk_rope_head_dim))], axis=-1)

    attn_bias = None
    attn_mask = None
    if mask is not None:
      if mask.dtype == jnp.bool_:
        attn_mask = mask
      else:
        # Treat the existing mask argument as an additive attention bias.
        if mask.ndim == 2:
          attn_bias = mask[None, None, :, :]
        else:
          attn_bias = mask

    out = jax.nn.dot_product_attention(
        q,
        k,
        v,
        bias=attn_bias,
        mask=attn_mask,
        scale=self.softmax_scale,
        is_causal=True,
        implementation="cudnn",
    )
    return self.wo(out.reshape(batch, seqlen, -1))


# Gate - Routes to the right MoE

class Gate(nn.Module):
  """
  Gate - Mechanism that routes to the right MoE Expert

  Routes tokens to top-k experts via 2-stage hierarchical selection:
    1. Score all experts via sigmoid/softmax
    2. Select top-k groups by top-2 expert score sum per group
    3. Select top-k experts from surviving groups
  Bias is per-expert for auxiliary-loss-free load balancing (DeepSeek V3).
  """
  dim: int
  n_routed_experts: int
  topk: int
  n_groups: int
  topk_groups: int
  route_scale: float
  score_func: str = 'sigmoid'



  @nn.compact
  def __call__(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:

    weight = self.param(
        'weight',
        nn.initializers.normal(stddev=0.01),
        (self.n_routed_experts, self.dim)
    )

    e_score_correction_bias = self.param(
        'e_score_correction_bias',
        nn.initializers.zeros,
        (self.n_routed_experts,)
    )

    scores = linear(x, weight.T)

    if self.score_func == "softmax":
      scores = jax.nn.softmax(scores, axis=-1).astype(jnp.float32)
    else:
      scores = jax.nn.sigmoid(scores)

    original_scores = scores

    # Add per-expert bias for routing decisions only
    scores = scores + e_score_correction_bias

    if self.n_groups > 1:
      scores = scores.reshape(x.shape[0], self.n_groups, -1)

      # Group score: sum of top-2 expert scores within each group
      group_score = jnp.sort(scores, axis=-1)[..., -2:].sum(axis=-1)

      group_idx = jnp.argsort(group_score, axis=-1)[..., -self.topk_groups:]

      mask = jnp.ones((x.shape[0], self.n_groups), dtype=bool)
      mask = mask.at[jnp.arange(x.shape[0])[:, None], group_idx].set(False)

      scores = jnp.where(mask[..., None], -jnp.inf, scores).reshape(x.shape[0], -1)

    # Select top-k experts
    indices = jnp.argsort(scores, axis=-1)[..., -self.topk:]

    # Final weights from original (un-biased) scores
    weights = original_scores[jnp.arange(x.shape[0])[:, None], indices]

    if self.score_func == "sigmoid":
      weights = weights / (weights.sum(axis=-1, keepdims=True) + 1e-20)

    weights = weights * self.route_scale

    return weights, indices


# Expert - The Expert in MoE

class Expert(nn.Module):
  dim: int
  inter_dim: int


  @nn.compact
  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """
    Forward pass for the Expert layer
    """
    w1 = Linear(self.dim, self.inter_dim)
    w2 = Linear(self.inter_dim, self.dim)
    w3 = Linear(self.dim, self.inter_dim)


    return w2(nn.silu(w1(x)) * w3(x))

# MLP - Mulit-Layer Perceptron


class MLP(nn.Module):
  dim: int
  inter_dim: int

  @nn.compact
  def __call__(self, x: jnp.ndarray):
    w1 = Linear(self.dim, self.inter_dim)
    w2 = Linear(self.inter_dim, self.dim)
    w3 = Linear(self.dim, self.inter_dim)

    return w2(jax.nn.silu(w1(x)) * w3(x))


# MoE - Mixture of Experts Layer

class MixtureOfExperts(nn.Module):

    dim: int
    n_routed_experts: int
    n_groups: int
    n_activated_experts: int
    n_shared_experts: int
    topk_groups: int
    inter_dim: int
    route_scale: float
    score_func: str


    def setup(self):
        self.gate = Gate(dim=self.dim, n_routed_experts=self.n_routed_experts,
                         topk=self.n_activated_experts,
                         n_groups=self.n_groups, topk_groups=self.topk_groups,
                         route_scale=self.route_scale, score_func=self.score_func
                         )
        # Vmapped expert: params stacked along axis 0, single call for all experts
        self.experts = nn.vmap(
            Expert,
            variable_axes={'params': 0},
            split_rngs={'params': True},
            in_axes=None, out_axes=0,
            axis_size=self.n_routed_experts
        )(self.dim, self.inter_dim)

        self.shared_experts = MLP(self.dim, self.n_shared_experts * self.inter_dim)

    def __call__(self, x: jnp.ndarray):
        shape = x.shape
        x = x.reshape(-1, self.dim)

        weights, indices = self.gate(x)

        # All experts in parallel: (n_experts, n_tokens, dim)
        all_expert_out = self.experts(x)

        # Gather selected expert outputs: (n_tokens, topk, dim)
        selected_out = all_expert_out[indices.T].transpose(1, 0, 2)  # TODO: verify indexing

        # Weighted sum over topk: (n_tokens, dim)
        y = (selected_out * weights[..., None]).sum(axis=1)

        z = self.shared_experts(x)
        return (y + z).reshape(shape)


# BottomBlock: KDA + MoE (3 of these per TopBlock)

class BottomBlock(nn.Module):
  """KDA_Layer followed by MoE_Layer"""
  dim: int
  heads: int
  head_dim: int
  kernel_size: int
  n_routed_experts: int
  n_groups: int
  n_activated_experts: int
  n_shared_experts: int
  topk_groups: int
  inter_dim: int
  route_scale: float
  score_func: str

  def setup(self):
    self.kda = KDA_Layer(dim=self.dim, heads=self.heads,
                         head_dim=self.head_dim, kernel_size=self.kernel_size)
    self.moe = MoE_Layer(dim=self.dim, n_routed_experts=self.n_routed_experts,
                         n_groups=self.n_groups, n_activated_experts=self.n_activated_experts,
                         n_shared_experts=self.n_shared_experts, topk_groups=self.topk_groups,
                         inter_dim=self.inter_dim, route_scale=self.route_scale,
                         score_func=self.score_func)

  def __call__(self, x: jnp.ndarray, kda_state: Optional[jnp.ndarray] = None):
    x, S = self.kda(x, kda_state)
    x = self.moe(x)
    return x, S


# TopBlock: MLA + MoE (1 of these per 3 BottomBlocks)

class TopBlock(nn.Module):
  """MLA_Layer followed by MoE_Layer"""
  dim: int
  heads: int
  local_heads: int
  q_lora_rank: int
  kv_lora_rank: int
  n_routed_experts: int
  n_groups: int
  n_activated_experts: int
  n_shared_experts: int
  topk_groups: int
  inter_dim: int
  route_scale: float
  score_func: str
  qk_head_dim: int = 128
  v_head_dim: int = 128

  def setup(self):
    self.mla = MLA_Layer(dim=self.dim, heads=self.heads,
                         local_heads=self.local_heads,
                         q_lora_rank=self.q_lora_rank,
                         kv_lora_rank=self.kv_lora_rank,
                         qk_head_dim=self.qk_head_dim,
                         v_head_dim=self.v_head_dim)
    self.moe = MoE_Layer(dim=self.dim, n_routed_experts=self.n_routed_experts,
                         n_groups=self.n_groups, n_activated_experts=self.n_activated_experts,
                         n_shared_experts=self.n_shared_experts, topk_groups=self.topk_groups,
                         inter_dim=self.inter_dim, route_scale=self.route_scale,
                         score_func=self.score_func)

  def __call__(self, x: jnp.ndarray, start: int = 0, freqs_cis: Optional[jnp.ndarray] = None, mask: Optional[jnp.ndarray] = None):
    x = self.mla(x, start, freqs_cis, mask)
    x = self.moe(x)
    return x



# KDA Layer

class KDA_Layer(nn.Module):
  """
  Kimi Delta Attention Layer

  Per-head pipeline: Linear proj -> ShortConv -> Swish -> L2Norm
  Then chunked delta attention via KDA, gated output projection.
  """
  dim: int
  heads: int
  head_dim: int = 128
  kernel_size: int = 4

  @nn.compact
  def __call__(self, x: jnp.ndarray, initial_state: Optional[jnp.ndarray] = None):
    batch, seq, dim = x.shape

    x_norm = RMSNorm(dim=self.dim)(x)

    # Per-head Q/K/V linear projections
    q_proj = nn.Dense(self.heads * self.head_dim, use_bias=False, name='wq')(x_norm)
    k_proj = nn.Dense(self.heads * self.head_dim, use_bias=False, name='wk')(x_norm)
    v_proj = nn.Dense(self.heads * self.head_dim, use_bias=False, name='wv')(x_norm)

    # Q: ShortConv -> Swish -> L2Norm
    q = ShortConv()(x=q_proj,
                    weights=self.param('q_conv_w', nn.initializers.lecun_normal(),
                                       (self.heads * self.head_dim, self.kernel_size)),
                    bias=self.param('q_conv_b', nn.initializers.zeros,
                                    (self.heads * self.head_dim,)),
                    k=self.kernel_size)
    q = nn.swish(q)
    q = q / (jnp.linalg.norm(q, axis=-1, keepdims=True) + 1e-6)

    # K: ShortConv -> Swish -> L2Norm
    k = ShortConv()(x=k_proj,
                    weights=self.param('k_conv_w', nn.initializers.lecun_normal(),
                                       (self.heads * self.head_dim, self.kernel_size)),
                    bias=self.param('k_conv_b', nn.initializers.zeros,
                                    (self.heads * self.head_dim,)),
                    k=self.kernel_size)
    k = nn.swish(k)
    k = k / (jnp.linalg.norm(k, axis=-1, keepdims=True) + 1e-6)

    # V: just activation, no conv/norm
    v = nn.swish(v_proj)

    # Reshape to (batch, seq, heads, head_dim)
    q = q.reshape(batch, seq, self.heads, self.head_dim)
    k = k.reshape(batch, seq, self.heads, self.head_dim)
    v = v.reshape(batch, seq, self.heads, self.head_dim)

    # Beta (per-head delta write gate)
    beta = nn.sigmoid(nn.Dense(self.heads, use_bias=False, name='w_beta')(x_norm))

    # Alpha (log-space decay gate) via low-rank projection
    alpha = nn.Dense(self.head_dim, use_bias=False, name='alpha_down')(x_norm)
    alpha = nn.Dense(self.heads * self.head_dim, use_bias=False, name='alpha_up')(alpha)
    alpha = nn.log_sigmoid(alpha)
    alpha = alpha.reshape(batch, seq, self.heads, self.head_dim)

    # Output gate via low-rank projection
    gate = nn.Dense(self.head_dim, use_bias=False, name='gate_down')(x_norm)
    gate = nn.Dense(self.heads * self.head_dim, use_bias=False, name='gate_up')(gate)
    gate = nn.sigmoid(gate)
    gate = gate.reshape(batch, seq, self.heads, self.head_dim)

    # Chunked delta attention
    kda_out, S = KDA(dim=self.dim, heads=self.heads, head_dim=self.head_dim)(
        q, k, v, alpha, beta, initial_state)

    # Gated output projection
    out = RMSNorm(dim=self.dim)(kda_out)
    out = out * gate
    out = out.reshape(batch, seq, self.heads * self.head_dim)
    final_out = nn.Dense(self.dim, use_bias=False, name='wo')(out)

    return x + final_out, S


# MoE Layer
class MoE_Layer(nn.Module):
  """
  Mixture of Experts Layer = RMSNorm + MoE + residual
  """
  dim: int
  n_routed_experts: int
  n_groups: int
  n_activated_experts: int
  n_shared_experts: int
  topk_groups: int
  inter_dim: int
  route_scale: float
  score_func: str

  def setup(self):
    self.norm = RMSNorm(self.dim)
    self.moe = MixtureOfExperts(dim=self.dim,
                                n_routed_experts=self.n_routed_experts,
                                n_groups=self.n_groups,
                                n_activated_experts=self.n_activated_experts,
                                n_shared_experts=self.n_shared_experts,
                                topk_groups=self.topk_groups,
                                inter_dim=self.inter_dim,
                                route_scale=self.route_scale,
                                score_func=self.score_func
                                )

  def __call__(self, x: jnp.ndarray):
    return x + self.moe(self.norm(x))


# MLA Layer

class MLA_Layer(nn.Module):
  """
  Multi-head Latent Attention Layer = RMSNorm + MLA + residual
  """
  dim: int
  heads: int
  local_heads: int
  q_lora_rank: int
  kv_lora_rank: int
  qk_head_dim: int = 128
  v_head_dim: int = 128

  def setup(self):
    self.norm = RMSNorm(self.dim)
    self.mla = multihead_attn(dim=self.dim,
                              heads=self.heads,
                              local_heads=self.local_heads,
                              q_lora_rank=self.q_lora_rank,
                              kv_lora_rank=self.kv_lora_rank,
                              qk_head_dim=self.qk_head_dim,
                              v_head_dim=self.v_head_dim)

  def __call__(self, x: jnp.ndarray, start: int = 0, freqs_cis: Optional[jnp.ndarray] = None, mask: Optional[jnp.ndarray] = None):
    return x + self.mla(self.norm(x), start, freqs_cis, mask)


# Kimi Linear Transformer: 3x BottomBlock (KDA+MoE) per 1x TopBlock (MLA+MoE)

class Transformer(nn.Module):
  """
  Kimi Linear Transformer

  Interleaves KDA and MLA layers in a 3:1 ratio, each followed by MoE.
  Pattern per group: [Bottom, Bottom, Bottom, Top] x n_groups
  """
  args: ModelArg

  def setup(self):
    # Shared MoE kwargs
    moe_kw = dict(
        n_routed_experts=self.args.n_routed_experts,
        n_groups=self.args.n_groups,
        n_activated_experts=self.args.n_activated_experts,
        n_shared_experts=self.args.shared_experts,
        topk_groups=self.args.topk_groups,
        inter_dim=self.args.inter_dim,
        route_scale=self.args.scaling_factor,
        score_func=self.args.score_func,
    )


    self.layers = []
    for g in range(self.args.n_layer_groups):
      for _ in range(3):
        self.layers.append(BottomBlock(
            dim=self.args.dim, heads=self.args.heads,
            head_dim=self.args.head_dim, kernel_size=4,
            **moe_kw))
      self.layers.append(TopBlock(
          dim=self.args.dim, heads=self.args.heads,
          local_heads=self.args.local_heads,
          q_lora_rank=self.args.q_lora_rank,
          kv_lora_rank=self.args.kv_lora_rank,
          qk_head_dim=self.args.qk_head_dim,
          v_head_dim=self.args.v_head_dim,
          **moe_kw))

    self.norm = RMSNorm(self.args.dim)
    self.tok_emb = nn.Embed(self.args.vocab_size, self.args.dim)
    self.head = nn.Dense(self.args.vocab_size, use_bias=False, name='lm_head')

  def __call__(self, tokens: jnp.ndarray, start: int = 0,
               freqs_cis: Optional[jnp.ndarray] = None,
               mask: Optional[jnp.ndarray] = None):
    x = self.tok_emb(tokens).astype(jnp.bfloat16)

    for layer in self.layers:
      if isinstance(layer, BottomBlock):
        x, _ = layer(x)
      else:
        x = layer(x, start, freqs_cis, mask)

    x = self.norm(x)
    # LM head in f32 for numerical precision in logits/loss
    return self.head(x.astype(jnp.float32))


# Training utils

def cross_entropy_loss(logits: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
  """Cross entropy in f32."""
  return optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()


@functools.partial(jax.jit, static_argnums=(0, 5))
def train_step(model, params, opt_state, tokens, labels, optimizer):
  """Single jitted training step.

  Args:
    model: Transformer module (static)
    params: model parameters pytree
    opt_state: optimizer state
    tokens: input token ids (int32)
    labels: target token ids (int32)
    optimizer: optax optimizer (static via closure or passed as arg)

  Returns:
    (loss, new_params, new_opt_state)
  """
  def loss_fn(params):
    logits = model.apply(params, tokens)
    return cross_entropy_loss(logits, labels)

  loss, grads = jax.value_and_grad(loss_fn)(params)
  updates, new_opt_state = optimizer.update(grads, opt_state, params)
  new_params = optax.apply_updates(params, updates)
  return loss, new_params, new_opt_state


@functools.partial(jax.jit, static_argnums=(0,))
def eval_step(model, params, tokens, labels):
  """Jitted evaluation step — forward pass + loss, no grads."""
  logits = model.apply(params, tokens)
  return cross_entropy_loss(logits, labels)


@jax.jit
def cosine_lr(step, max_lr, min_lr, warmup_steps, total_steps):
    warmup_lr = max_lr * (step / warmup_steps)

    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    cosine_lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + jnp.cos(jnp.pi * progress))

    return jnp.where(step < warmup_steps, warmup_lr, cosine_lr)


# Training

class Training:
    """
    Baseline Kimi Linear training loop.
    """

    def setup(self):
        pass


    def train(self, model, optimizer, train_ds, eval_ds, max_steps,):

        for _ in range(max_steps):
            pass


        pass
