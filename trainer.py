
import jax
import optax
import functools
import jax.numpy as jnp
import huggingface_hub


# Utils


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




class Training:
    """
    Training clas for three models:

    1. Baseline Kimi Linear
    2. Chunked Attention Residual (Kimi Linear)
    3. Full Attention Residual (Kimi Linear)
    """
    def setup(self):

        pass
    def train(self):
        pass