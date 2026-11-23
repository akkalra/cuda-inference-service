"""Tensor-parallel helpers (NCCL on CUDA, ``neuron`` on Trainium, ``gloo`` on CPU)."""
import os

import torch
import torch.distributed as dist


def init_distributed(backend=None):
    """Initialise the process group when launched under torchrun; no-op for single-process runs."""
    if dist.is_initialized() or "RANK" not in os.environ:
        return 0, 1
    from inference.backends import get_backend

    backend = get_backend(backend)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    backend.set_local_device(local_rank)
    dist.init_process_group(backend=backend.distributed_backend())
    return dist.get_rank(), dist.get_world_size()


def all_reduce_logits(logits: torch.Tensor) -> torch.Tensor:
    if dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(logits, op=dist.ReduceOp.SUM)
    return logits
