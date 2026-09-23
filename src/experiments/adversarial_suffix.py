"""Controlled adversarial-suffix research scaffold.

This module demonstrates the differentiable part of a gradient-based suffix
experiment without shipping harmful targets or a ready-made jailbreak recipe.

Use only synthetic/benign targets in an authorized research environment.
"""

import torch
import torch.nn.functional as F


def target_loss(
    model,
    prompt_embeds: torch.Tensor,
    suffix_ids: torch.Tensor,
    target_ids: torch.Tensor,
    embedding_layer,
) -> torch.Tensor:
    """Calculate teacher-forced target-token cross-entropy."""
    suffix_embeds = embedding_layer(suffix_ids)

    full_embeds = torch.cat(
        [
            prompt_embeds,
            suffix_embeds,
            embedding_layer(target_ids).detach(),
        ],
        dim=1,
    )

    outputs = model(
        inputs_embeds=full_embeds,
        use_cache=False,
    )

    prefix_len = prompt_embeds.shape[1] + suffix_ids.shape[1]
    target_len = target_ids.shape[1]

    target_logits = outputs.logits[
        :,
        prefix_len - 1: prefix_len - 1 + target_len,
        :,
    ]

    return F.cross_entropy(
        target_logits.reshape(-1, target_logits.size(-1)),
        target_ids.reshape(-1),
    )


def candidate_token_ids(
    suffix_grad: torch.Tensor,
    embedding_layer,
    position: int,
    top_k: int = 10,
) -> torch.Tensor:
    """Return token IDs aligned with the negative gradient direction."""
    grad = suffix_grad[0, position]

    scores = -torch.matmul(
        embedding_layer.weight,
        grad,
    )

    _, token_ids = torch.topk(scores, k=top_k)
    return token_ids


def freeze_model(model) -> None:
    """Freeze model parameters for input-optimization experiments."""
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
