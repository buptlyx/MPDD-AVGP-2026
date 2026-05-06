from __future__ import annotations

import logging
from collections.abc import Callable

import torch


CUDA_FALLBACK_INDICES = (0, 1, 2, 3)


def is_cuda_oom(error: BaseException) -> bool:
    message = str(error).lower()
    return isinstance(error, RuntimeError) and "cuda" in message and "out of memory" in message


def clear_cuda_cache() -> None:
    if not torch.cuda.is_available():
        return
    torch.cuda.empty_cache()
    try:
        torch.cuda.ipc_collect()
    except RuntimeError:
        pass


def _parse_cuda_index(device_name: str) -> int | None:
    if device_name == "cuda":
        return 0
    if not device_name.startswith("cuda:"):
        return None
    try:
        return int(device_name.split(":", 1)[1])
    except ValueError:
        return None


def resolve_device_candidates(requested_device: str) -> list[torch.device]:
    requested_device = str(requested_device).strip().lower()
    if not requested_device.startswith("cuda"):
        return [torch.device(requested_device)]

    if not torch.cuda.is_available():
        return [torch.device("cpu")]

    visible_count = torch.cuda.device_count()
    requested_index = _parse_cuda_index(requested_device)
    if requested_index is None:
        preferred_indices = list(CUDA_FALLBACK_INDICES)
    else:
        preferred_indices = [requested_index] + [idx for idx in CUDA_FALLBACK_INDICES if idx != requested_index]

    candidates = [torch.device(f"cuda:{idx}") for idx in preferred_indices if idx < visible_count]
    return candidates or [torch.device("cpu")]


def build_model_on_available_device(
    model_factory: Callable[[], torch.nn.Module],
    requested_device: str,
    logger: logging.Logger | None = None,
    preflight: Callable[[torch.nn.Module, torch.device], None] | None = None,
) -> tuple[torch.nn.Module, torch.device]:
    candidates = resolve_device_candidates(requested_device)
    oom_messages: list[str] = []

    for device in candidates:
        model = model_factory()
        try:
            model = model.to(device)
            if preflight is not None:
                preflight(model, device)
        except RuntimeError as error:
            if device.type == "cuda" and is_cuda_oom(error):
                oom_messages.append(f"{device}: {error}")
                if logger is not None:
                    logger.warning("CUDA OOM on %s, trying next device.", device)
                del model
                clear_cuda_cache()
                continue
            raise

        if logger is not None and str(device) != str(candidates[0]):
            logger.info("Using fallback device: %s", device)
        return model, device

    joined = "\n".join(oom_messages)
    raise RuntimeError(f"All CUDA fallback devices ran out of memory:\n{joined}")
