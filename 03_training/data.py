import torch


def get_batch(
    data: torch.Tensor,
    batch_size: int,
    context_length: int,
    device: str | torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if data.ndim != 1:
        raise ValueError(f"Expected 1D token data, got shape {tuple(data.shape)}")

    if len(data) <= context_length:
        raise ValueError(
            f"Data length must be greater than context_length. "
            f"Got len(data)={len(data)}, context_length={context_length}."
        )

    start_indices = torch.randint(
        low=0,
        high=len(data) - context_length,
        size=(batch_size,),
        device=data.device,
    )

    offsets = torch.arange(context_length, device=data.device)
    batch_indices = start_indices[:, None] + offsets[None, :]

    x = data[batch_indices]
    y = data[batch_indices + 1]

    return x.to(device), y.to(device)
