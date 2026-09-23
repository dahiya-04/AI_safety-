"""Token-level likelihood utilities."""

import torch
import torch.nn.functional as F

from .model_loader import load_model


def get_answer_logprob(
    model,
    tokenizer,
    question: str,
    option: str,
) -> float:
    """Return mean log probability of the candidate answer tokens."""
    prefix = f"Question: {question}\nAnswer:"
    full_text = prefix + " " + option

    prefix_ids = tokenizer(
        prefix,
        return_tensors="pt",
        add_special_tokens=True,
    ).input_ids.to(model.device)

    full_ids = tokenizer(
        full_text,
        return_tensors="pt",
        add_special_tokens=True,
    ).input_ids.to(model.device)

    with torch.no_grad():
        logits = model(full_ids).logits

    shift_logits = logits[:, :-1, :]
    shift_labels = full_ids[:, 1:]
    log_probs = F.log_softmax(shift_logits, dim=-1)

    prefix_len = prefix_ids.shape[1]
    answer_scores = []

    for position in range(prefix_len - 1, full_ids.shape[1] - 1):
        token_id = shift_labels[0, position]
        answer_scores.append(log_probs[0, position, token_id])

    if not answer_scores:
        raise ValueError("The candidate answer contains no scored tokens.")

    return torch.stack(answer_scores).mean().item()


def rank_options(model, tokenizer, question: str, options: list[str]):
    """Rank candidate answers by their mean token log probability."""
    scored = [
        (option, get_answer_logprob(model, tokenizer, question, option))
        for option in options
    ]
    return sorted(scored, key=lambda item: item[1], reverse=True)


if __name__ == "__main__":
    bundle = load_model()

    question = "What is the capital of France?"
    options = ["Berlin", "Madrid", "Paris", "Rome"]

    for option, score in rank_options(
        bundle.model,
        bundle.tokenizer,
        question,
        options,
    ):
        print(f"{option}: {score:.4f}")
