import copy
from typing import Literal

import torch
from torch import nn

import torch.nn.functional as F

from main.model.VATE.constrastive_model import MaskedContrastiveModel, MaskedContrastiveModelOutputs
from main.model.blocks.time_masked import TimeMaskSwitchableProperties
from main.model.neegavi.model import EegInterAviModel
from main.model.neegavi.train_utils import KdTrainDataModule
from main.model.neegavi.trainer.default import EegAviKdVateMaskedSemiSupervisedModule
from main.model.neegavi.utils import WeaklySupervisedEegBaseModelOutputs
from main.utils.data import MaskedValue


class MocoStyleNEegAviTrainer(EegAviKdVateMaskedSemiSupervisedModule):

    def __init__(
            self,
            student: EegInterAviModel,
            teacher: MaskedContrastiveModel,
            datamodule: KdTrainDataModule,
            dequantize_keys: list[str],
            kd_loss_weight: float,
            fusion_loss_weight: float,
            fusion_metrics: list[str],
            kd_keys: list[str],
            lr: float,
            kd_temperature: float,
            bidirectional_p: float = .9,
            seed: int = 1,
            batch_size=None,
            momentum: float = .9,
            queue_size: int = 512
    ):
        super().__init__(student, teacher, datamodule, dequantize_keys, kd_loss_weight, fusion_loss_weight,
                         fusion_metrics, kd_keys, lr, kd_temperature, bidirectional_p, seed, batch_size)

        self.momentum_student = copy.deepcopy(student)
        for p in self.momentum_student.parameters():
            p.requires_grad_(False)

        self.register_buffer("queue_ptr", torch.zeros((), dtype=torch.long))
        self.moco_queues = nn.ModuleDict()  # store buffers by name is annoying; use dict of tensors instead
        self.moco_queue = {}  # key -> tensor [K, D]

        self.momentum: float = momentum
        self.queue_size: int = queue_size

    @torch.no_grad()
    def moco_momentum_update(self):
        m = self.momentum
        for model_parameter, momentum_parameter in zip(self.student.parameters(), self.momentum_student.parameters()):
            momentum_parameter.mul_(model_parameter).add_(model_parameter.data, alpha=1. - m)

    def state_update(self):
        super().state_update()
        self.moco_momentum_update()

    @torch.no_grad()
    def moco_init_queue(self, key: str, dim: int, device):
        if key not in self.moco_queue:
            self.moco_queue[key] = torch.zeros(self.queue_size, dim, device=device)
            # Optional: normalize storage
            self.moco_queue[key] = F.normalize(self.moco_queue[key], dim=-1)

    def training_step(self, batch, batch_idx):
        mode = self.state_update()
        # Convert the batch to fp16 from quantized
        batch = self.dequantize(self.nest(batch), torch.float16)
        stud_out: WeaklySupervisedEegBaseModelOutputs = self.student(batch["student"], use_kd=True)

        self.momentum_student.set_attention_modality(TimeMaskSwitchableProperties(mode=mode))
        with torch.no_grad():
            momentum_out = self.momentum_student(batch["student"], use_kd=True)
            # todo verifcare sta merda
            self.mom_modality_pooled = {}
            for key, mv in momentum_out.multimodal_outs.items():
                valid = self._get_y_valid(mv)
                if valid.any():
                    self.mom_modality_pooled[key] = self._y_mean(mv, valid)

        with torch.inference_mode():
            teacher_out: MaskedContrastiveModelOutputs = self.teacher(batch["teacher"])

        return self._compute_step_metrics(stud_out, teacher_out, batch, 'train', mode)

    # todo studia
    # Note: this uses one shared pointer for all queues; that’s fine if you enqueue the same batch count per key.
    # If not, give each key its own pointer
    @torch.no_grad()
    def moco_enqueue(self, key: str, x: torch.Tensor):
        if x.size(0) > self.queue_size:
            self.moco_queue[key].copy_(x[-self.queue_size:])
            self.queue_ptr.fill_(0)

            return

        ptr = self.queue_ptr.item()
        end = ptr + x.size(0)  # b

        if end <= self.queue_size:
            self.moco_queue[key][ptr:end] = x
        else:
            first = self.queue_size - ptr
            self.moco_queue[key][ptr:] = x[:first]
            self.moco_queue[key][:end - self.queue_size] = x[first:]

        self.queue_ptr.fill_((ptr + x.size(0)) % self.queue_size)

    def compute_fusion_loss(self, fused_output: torch.Tensor, modality_outputs: dict[str, MaskedValue],
                            step_type: Literal['train', 'val', 'test'],
                            mode: Literal['bidirectional', 'causal']) -> torch.Tensor:
        base_loss = torch.tensor(0.0, device=fused_output.device)
        count_present = 0

        for key, value in modality_outputs.items():
            self.verbose and self.inner_logger.info(f"\nFor key {key}:")
            # Invalid rows are discarded
            valid_rows = self._get_y_valid(value)
            if not valid_rows.any():
                continue
            z_pos = self._y_mean(value, valid_rows)

            # Add the moco elements
            b_queue = None
            if step_type == "train":
                self.moco_init_queue(key, dim=z_pos.size(-1), device=z_pos.device)
                b_queue = self.moco_queue[key]

            count_present += 1
            mod_loss = self.siglip_losses[key](fused_output[valid_rows], self._y_mean(value, valid_rows), b_queue)

            self.log(f"{step_type}/fusion/{mode}/{key}", mod_loss, on_epoch=True, on_step=True, prog_bar=False)
            base_loss = base_loss + mod_loss

            if step_type == "train":
                # enqueue momentum keys (must correspond to the same rows used for loss)
                k_all = self._mom_modality_pooled.get(key, None)
                if k_all is not None:
                    self.moco_enqueue(key, k_all)  # consider enqueueing only valid rows if needed

        return base_loss / count_present
