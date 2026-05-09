import torch


def cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    logits:  (batch_size, seq_len, vocab_size)
    targets: (batch_size, seq_len)

    return: scalar loss
    """

    batch_size, seq_len, vocab_size = logits.shape

    # Flatten sequence positions into independent next-token predictions.
    logits_flat = logits.view(batch_size * seq_len, vocab_size)
    targets_flat = targets.view(batch_size * seq_len)

    logsumexp = torch.logsumexp(logits_flat, dim=-1)

    correct_logits = logits_flat[
        torch.arange(batch_size * seq_len, device=logits.device),
        targets_flat,
    ]

    loss = logsumexp - correct_logits

    return loss.mean()
