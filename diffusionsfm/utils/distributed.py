import os
import socket
from contextlib import closing

import torch.distributed as dist


def get_open_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


# Distributed process group
def ddp_setup(rank, world_size, port="12345"):
    """
    Args:
        rank: Unique Identifier
        world_size: number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    print(f"MasterPort: {str(port)}")
    os.environ["MASTER_PORT"] = str(port)

    # initialize the process group
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()
