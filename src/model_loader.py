"""Model loading utilities for AAIGM."""

from dataclasses import dataclass
from typing import Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class ModelBundle:
    model: AutoModelForCausalLM
    tokenizer: AutoTokenizer


def load_model(
    model_name: str = "Qwen/Qwen3-1.7B",
) -> ModelBundle:
    """Load a causal language model and tokenizer.

    The model is automatically placed on available hardware using
    Transformers' device_map mechanism.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )

    model.eval()
    return ModelBundle(model=model, tokenizer=tokenizer)


def environment_info() -> dict:
    """Return basic runtime information."""
    return {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
    }


if __name__ == "__main__":
    print(environment_info())
