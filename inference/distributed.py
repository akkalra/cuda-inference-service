"""Tensor-parallel helpers (NCCL)."""
import os

import torch
import torch.distributed as dist


def init_distributed():
    if dist.is_initialized() or "RANK" not in os.environ:
        return 0, 1
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return dist.get_rank(), dist.get_world_size()


def all_reduce_logits(logits: torch.Tensor) -> torch.Tensor:
    if dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(logits, op=dist.ReduceOp.SUM)
    return logits
