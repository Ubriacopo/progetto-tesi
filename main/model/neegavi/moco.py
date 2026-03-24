from abc import abstractmethod, ABC

import lightning
import torch
from torch.nn import functional as F


class MoCoAble(ABC, lightning.LightningModule):
    def __init__(self, use_moco: bool, momentum: float = .99, queue_size: int = 512, ):
        super().__init__()
        # If MoCo style training is enabled
        self.use_moco: bool = use_moco
        self.momentum: float = momentum
        self.queue_size: int = queue_size

        self.moco_queue = {}  # key -> tensor [K, D]
        self.queue_ptr = {}  # key -> 0-dim long buffer

    @abstractmethod
    def moco_momentum_update(self):
        pass

    @torch.no_grad()
    def moco_init_queue(self, key: str, dim: int, device):
        if key not in self.moco_queue:
            q = F.normalize(torch.randn(self.queue_size, dim, device=device), dim=-1)

            self.register_buffer(f"moco_queue_{key}", q)
            self.register_buffer(f"queue_ptr_{key}", torch.zeros((), dtype=torch.long, device=device))

            self.moco_queue[key] = getattr(self, f"moco_queue_{key}")
            self.queue_ptr[key] = getattr(self, f"queue_ptr_{key}")

    @torch.no_grad()
    def moco_enqueue(self, key: str, x: torch.Tensor):
        x = F.normalize(x.detach(), dim=-1)
        ptr_buf = self.queue_ptr[key]

        if x.size(0) > self.queue_size:
            self.moco_queue[key].copy_(x[-self.queue_size:])
            ptr_buf.fill_(0)
            return

        ptr = int(ptr_buf.item())
        end = ptr + x.size(0)  # b

        if end <= self.queue_size:
            self.moco_queue[key][ptr:end] = x
        else:
            first = self.queue_size - ptr
            self.moco_queue[key][ptr:] = x[:first]
            self.moco_queue[key][:end - self.queue_size] = x[first:]

        ptr_buf.fill_((ptr + x.size(0)) % self.queue_size)

    def optimizer_step(self, epoch: int, batch_idx: int, optimizer, optimizer_closure=None, **kwargs) -> None:
        # Let Lightning handle closure semantics properly
        optimizer.step(closure=optimizer_closure)
        # Update EMA after the actual optimizer step
        if self.use_moco:
            self.moco_momentum_update()
