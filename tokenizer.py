from typing import TYPE_CHECKING, Any

import jax.numpy as jnp

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase
else:
    PreTrainedTokenizerBase = Any


class TextTokenizer:
    def __init__(self, tokenizer_name: str = "gpt2", max_length: int = 4097):


        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            tokenizer_name,
            use_fast=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_length = max_length

    @property
    def vocab_size(self) -> int:
        return len(self.tokenizer)

    def encode_batch(self, texts: list[str]) -> dict[str, jnp.ndarray]:
        encoded = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="np",
        )

        input_ids = jnp.asarray(encoded["input_ids"], dtype=jnp.int32)
        attention_mask = jnp.asarray(encoded["attention_mask"], dtype=jnp.float32)

        return {
            "tokens": input_ids[:, :-1],
            "labels": input_ids[:, 1:],
            "loss_mask": attention_mask[:, 1:],
        }
