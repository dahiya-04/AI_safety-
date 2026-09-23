"""Text generation utilities."""

import argparse
from typing import Optional

import torch

from .model_loader import load_model


def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 200,
    do_sample: bool = False,
) -> str:
    """Generate text and return only newly generated tokens."""
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
    else:
        text = prompt

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = output_ids[0, inputs.input_ids.shape[-1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument(
        "--prompt",
        default="Explain why reproducible machine-learning experiments are important.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=150)
    args = parser.parse_args()

    bundle = load_model(args.model)
    answer = generate_text(
        bundle.model,
        bundle.tokenizer,
        args.prompt,
        max_new_tokens=args.max_new_tokens,
    )

    print(answer)


if __name__ == "__main__":
    main()
