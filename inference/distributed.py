"""Process-group setup for tensor parallelism (NCCL on CUDA, XLA on Neuron, Gloo on CPU)."""
import os

import torch
import torch.distributed as dist


def init_distributed(device: str = "cuda"):
    if dist.is_initialized() or "RANK" not in os.environ:
        return 0, 1
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if device == "cuda":
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
    elif device == "neuron":
        # Registers the "xla" backend; collectives lower to Neuron collective-communication ops
        # over the NeuronLink/EFA fabric. Launch with torchrun as on GPUs.
        import torch_xla.distributed.xla_backend  # noqa: F401

        dist.init_process_group(backend="xla", init_method="xla://")
    else:
        dist.init_process_group(backend="gloo")
    return dist.get_rank(), dist.get_world_size()


def all_reduce_logits(logits: torch.Tensor) -> torch.Tensor:
    if dist.is_initialized() and dist.get_world_size() > 1:
        if dist.get_backend() == "xla" and logits.device.type != "xla":
            import torch_xla

            reduced = logits.to(torch_xla.device())
            dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
            return reduced.cpu()
        dist.all_reduce(logits, op=dist.ReduceOp.SUM)
    return logits
