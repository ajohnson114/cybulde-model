import os

from contextlib import contextmanager
from typing import Generator

import torch


def get_local_rank() -> int:
    return int(os.getenv("LOCAL_RANK", -1))


def get_global_rank() -> int:
    return int(os.getenv("RANK", get_local_rank()))


@contextmanager
def local_rank_zero_first() -> Generator[None, None, None]:
    if not torch.distributed.is_initialized() and os.getenv("RANK") is not None:
        raise RuntimeError("RANK is set but torch.distributed is not initialized")

    if torch.distributed.is_initialized():
        rank = get_local_rank()
        if rank not in [-1, 0]:
            torch.distributed.barrier()  # type: ignore
        yield
        if rank == 0:
            torch.distributed.barrier()  # type: ignore
    else:
        yield


@contextmanager
def global_rank_zero_first() -> Generator[None, None, None]:
    """
    Forces all processes to wait for the global rank zero to finish the task
    and from that point on will allow them to finish. In simpler terms,
    in a distributed system you have worker nodes and a coordinating node and the
    coordinating node is referred to as the global_rank_zero. So what's happening 
    here is that we want to make sure that when we export our model it's finalized at
    that step so we wait for the coordinating node to finish running the task and then
    have all the workers execute the task so it can see the final model before exporting. 
    """

    if not torch.distributed.is_initialized() and os.getenv("RANK") is not None:
        raise RuntimeError("RANK is set but torch.distributed is not initialized")

    if torch.distributed.is_initialized():
        rank = get_global_rank()
        if rank not in [-1, 0]: #if not 0 or not using torch.distributed wait
            torch.distributed.barrier()  # type: ignore
        yield #this and above makes the coordinating node yield first
        if rank == 0: #this makes the coordinating node wait for the others
            torch.distributed.barrier()  # type: ignore
            #torch.distributed.barrier makes sure that all the ranks are synched
    else:
        yield