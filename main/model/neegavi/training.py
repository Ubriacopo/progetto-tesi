import copy
import math
from typing import Optional, Literal, Any

import lightning as L
import torch
from lightning.pytorch.utilities.types import OptimizerLRScheduler, STEP_OUTPUT
from torch import nn
from torch.nn import functional as F
from torch.optim import Optimizer

from main.model.VATE.constrastive_model import MaskedContrastiveModel, MaskedContrastiveModelOutputs
from main.model.blocks.pooling import ClsPooling, MaskedAvgPooling
from main.model.blocks.time_masked import TimeMaskSwitchableProperties
from main.model.loss import SiglipLoss
from main.model.neegavi.model import EegInterAviModel
from main.model.neegavi.train_utils import KdTrainDataModule
from main.model.neegavi.utils import EegBaseModelOutputs, WeaklySupervisedEegBaseModelOutputs
from main.utils.data import MaskedValue
from main.utils.logging import make_logger


class EegAviKdVateMaskedSemiSupervisedModule(L.LightningModule):
    FUSED_KEY: str = "fused"
    PIVOT_KEY: str = 'eeg'

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
            bidirectional_p: float = .9,  # For ATTN
            seed: int = 1,
            use_moco: bool = False,
            momentum: float = .9,
            queue_size: int = 104,
            batch_size=None
    ):
        super().__init__()
        self.batch_size = batch_size
        self.datamodule: KdTrainDataModule = datamodule

        self.inner_logger = make_logger(self.__class__.__name__)
        self.verbose: bool = False

        self.student = student
        self.teacher = teacher

        self.siglip_losses: nn.ModuleDict = nn.ModuleDict()
        for fusion_metric in fusion_metrics:
            loss_fn = SiglipLoss(init_tau=0.07, init_bias=-10, stop_grad_target=True, verbose=self.verbose)
            self.siglip_losses.add_module(fusion_metric, loss_fn)

        self.kd_losses: nn.ModuleDict = nn.ModuleDict()
        for kd_key in kd_keys:
            loss_fn = SiglipLoss(init_tau=0.05, init_bias=-10, stop_grad_target=True)
            self.kd_losses.add_module(kd_key, loss_fn.to(self.device))

        # Just to debug atm
        self.use_kd_loss = True
        self.use_fusion_loss = True

        self.base_seed = seed
        self.time_mask_switch_generator = torch.Generator(device=self.device)
        self.time_mask_switch_generator.manual_seed(seed)
        self.bidirectional_p: float = bidirectional_p

        self.dequantize_keys: list[str] = dequantize_keys

        # Hyperparameters
        self.lr: float = lr
        self.kd_temperature: float = kd_temperature
        # Weights of different losses combined
        self.alpha: float = kd_loss_weight
        self.beta: float = fusion_loss_weight

        self.k: int = 5

        # Utils
        self._n_causal: int = 0
        self._n_bidirectional: int = 0

        # If MoCo style training is enabled
        self.use_moco: bool = use_moco
        self.momentum_modality_pooled: Optional[dict[str, torch.Tensor]] = None
        self.momentum_student: Optional[EegInterAviModel] = None
        # Even if we don't use moco we initialize these for practicality
        self.momentum: float = momentum
        self.queue_size: int = queue_size
        self.moco_queue = {}  # key -> tensor [K, D]
        self.queue_ptr = {}  # key -> 0-dim long buffer

        if self.use_moco:
            # TODO Verifica deep copy
            self.momentum_student = copy.deepcopy(self.student)
            for parameter in self.momentum_student.parameters():
                parameter.requires_grad_(False)

    # TODO:
    # Okay so hear me out: First model I do without MoCo I then do reverse ablation and add MoCo and see if it helps? yes
    def configure_optimizers(self) -> OptimizerLRScheduler:
        params = []

        params += [
            # siglip_common_optim_configs for Fusion
            {"params": i.parameters(), "lr": self.lr * 5, "weight_decay": 0.0}
            for i in self.siglip_losses.values()
        ]

        params += [
            # siglip_common_optim_configs for KD
            {"params": i.parameters(), "lr": self.lr * 5, "weight_decay": 0.0}
            for i in self.kd_losses.values()
        ]

        params += [{"params": self.student.parameters(), "lr": self.lr}]  # Student parameters
        return torch.optim.Adam(weight_decay=.01, params=params, fused=True)

    def _compute_step_metrics(
            self,
            stud: EegBaseModelOutputs,
            teacher: MaskedContrastiveModelOutputs,
            batch,
            step_type: Literal['train', 'val', 'test'],
            mode: Literal['bidirectional', 'causal']
    ):
        return_object: dict[str, torch.Tensor | MaskedValue] = dict(
            loss=torch.tensor(0, device=stud.embeddings['data'].device)
        )

        if self.use_kd_loss:
            kd_loss = self.compute_kd_loss(student_out=stud.kd_outs, teacher_out=teacher, step_type=step_type)
            return_object["loss"] = return_object["loss"] + kd_loss * self.alpha

        if self.use_fusion_loss:
            fusion_loss = self.compute_fusion_loss(
                fused_output=stud.cls, modality_outputs=stud.multimodal_outs, step_type=step_type, mode=mode
            )

            return_object["loss"] = return_object["loss"] + fusion_loss * self.beta
            self.log(f"{step_type}/fusion", fusion_loss, on_epoch=False, on_step=True, prog_bar=True)

            # For later evaluations
            return_object[self.FUSED_KEY] = stud.cls.detach()
            for key, masked_value in stud.multimodal_outs.items():
                masked_value["data"] = masked_value["data"].detach()
                masked_value["mask"] = masked_value["mask"].detach()
                return_object[key] = masked_value

        train = step_type == "train"
        self.log(f"{step_type}/loss", return_object["loss"], prog_bar=train, on_step=True, on_epoch=True)
        self.log(f"{step_type}/loss-{mode}", return_object["loss"], prog_bar=True, on_step=True, on_epoch=False)
        return return_object

    @torch.no_grad()
    def moco_momentum_update(self):
        m = self.momentum
        for model_parameter, momentum_parameter in zip(self.student.parameters(), self.momentum_student.parameters()):
            momentum_parameter.data.mul_(m).add_(model_parameter.data, alpha=1. - m)

    @torch.no_grad()
    def moco_init_queue(self, key: str, dim: int, device):
        if key not in self.moco_queue:
            self.moco_queue[key] = F.normalize(torch.randn(self.queue_size, dim, device=device), dim=-1)
            self.register_buffer(f"queue_ptr_{key}", torch.zeros((), dtype=torch.long))
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

    @staticmethod
    # todo move
    def _topk_hits_from_sim(sim: torch.Tensor, ks: tuple[int, ...]) -> dict[int, torch.Tensor]:
        # sim: (n, n)
        n = sim.size(0)
        device = sim.device
        gt = torch.arange(n, device=device)

        out = {}
        kmax = min(max(ks), sim.size(1))
        top = sim.topk(kmax, dim=1).indices  # (n, kmax)

        # compare once
        eq = top.eq(gt[:, None])  # (n, kmax)
        for k in ks:
            k = min(k, sim.size(1))
            out[k] = eq[:, :k].any(dim=1).float().mean()

        return out

    # todo expensive only on some batches
    def _compute_batch_metrics(self, fused_z, pivot_z, outputs: dict, step_type: Literal['train', 'val', 'test'],
                               mode: Optional[Literal['bidirectional', 'causal']] = None):
        # Euclidean distance between the two embedding spaces
        dist = (pivot_z - fused_z).norm(dim=1).mean()
        on_step = step_type == 'train'
        on_epoch = not on_step

        self.log(f"{step_type}/norm(eeg-fused)", dist, on_step=on_step, on_epoch=on_epoch)
        prefix = "" if mode is None else f"{mode}/"

        fused = F.normalize(fused_z, dim=-1)
        pivot = F.normalize(pivot_z, dim=-1)

        top_k_values = (1, 3, self.k)
        top1_mean = []
        for key, embedding in outputs.items():
            valid = self._get_y_valid(embedding)
            if not valid.any():
                continue  # This modality cannot be evaluated

            e = self._y_mean(embedding, valid)
            e = F.normalize(e, dim=-1)  # Normalize once
            f, p = fused[valid], pivot[valid]

            sim_fe = f @ e.T
            hits_fe = self._topk_hits_from_sim(sim_fe, top_k_values)
            hits_ef = self._topk_hits_from_sim(sim_fe.T, top_k_values)  # reuse transpose
            pre_f = f"{step_type}/{prefix}fused/"
            # TOP-1 FUSED
            self.log(f"{pre_f}top1_{key}", hits_fe[1], on_step=on_step, on_epoch=on_epoch)
            self.log(f"{step_type}/fused/top1_{key}", hits_fe[1], on_step=on_step, on_epoch=on_epoch)
            self.log(f"{pre_f}top1_{key}_R", hits_ef[1], on_step=on_step, on_epoch=on_epoch)
            self.log(f"{pre_f}top3_{key}", hits_fe.get(3, torch.nan), on_step=on_step, on_epoch=on_epoch)
            self.log(f"{step_type}/fused/top3_{key}", hits_fe.get(3, torch.nan), on_step=on_step, on_epoch=on_epoch)
            self.log(f"{pre_f}top3_{key}_R", hits_ef.get(3, torch.nan), on_step=on_step, on_epoch=on_epoch)
            self.log(f"{pre_f}top{self.k}_{key}", hits_fe.get(self.k, torch.nan), on_step=on_step, on_epoch=on_epoch)
            self.log(f"{step_type}/fused/top{self.k}_{key}", hits_fe.get(self.k, torch.nan), on_step=on_step,
                     on_epoch=on_epoch)
            self.log(f"{pre_f}top{self.k}_{key}_R", hits_ef.get(self.k, torch.nan), on_step=on_step, on_epoch=on_epoch)
            top1_mean.append(hits_fe[1])

            # TODO: Io qui vado avanti perche semplicmente le metriche dopo sono solo noisy (in speranza di risparmaire tempo)
            if key == self.PIVOT_KEY or True:
                continue
            # pivot <-> emb: one matmul
            sim_pe = p @ e.T
            hits_pe = self._topk_hits_from_sim(sim_pe, (1,))
            hits_ep = self._topk_hits_from_sim(sim_pe.T, (1,))

            self.log(f"{step_type}/{prefix}pivot/top1_{key}", hits_pe[1], on_step=on_step, on_epoch=on_epoch)
            self.log(f"{step_type}/{prefix}pivot/top1_{key}_R", hits_ep[1], on_step=on_step, on_epoch=on_epoch)
            delta = hits_fe[1] - hits_pe[1]
            self.log(f"{step_type}/{prefix}delta_{key}", delta, on_step=on_step, on_epoch=on_epoch)

        top1_mean = torch.mean(torch.stack(top1_mean, dim=0))
        self.log(f"{step_type}/top1_mean", top1_mean, on_step=on_step, on_epoch=on_epoch)

    def p_causal_schedule(self, start: float = .05, end: float = .8, floor_bidirectional: float = .1):
        """
        Common practice: Current setups favors bidirectional at lower epochs and causal later ones.
        AntLM (2024): explicitly describes a unified framework that alternates/switches between causal LM (causal mask) and masked LM (bidirectional attention).
        :param step:
        :param max_steps:
        :param start:
        :param end:
        :param floor_bidirectional:
        :return:
        """
        t = min(self.trainer.global_step / self.trainer.max_steps, 1.0)
        # Cosine ramp
        p = start + 0.5 * (end - start) * (1 - math.cos(math.pi * t))
        return min(p, 1.0 - floor_bidirectional)

    def on_train_start(self) -> None:
        self.time_mask_switch_generator = torch.Generator(device=self.device)
        self.time_mask_switch_generator.manual_seed(self.base_seed)

    def dequantize(self, batch: dict, dtype=torch.float16):
        # Student part:
        with torch.profiler.record_function("dequantize"):
            output = {}
            for container_key, container in batch.items():  # Student - Teacher
                output[container_key] = {}

                for key, td in container.items():
                    if key in self.dequantize_keys:
                        data = td["data"].to(dtype=dtype, non_blocking=True)
                        data.mul_(td["scales"])  # For optimization reasons (I dislike it)
                        td = {"data": data, "mask": td["mask"]}
                    output[container_key][key] = td

        return output

    @staticmethod
    def nest(flat):
        root = {}
        for key, value in flat.items():
            parts = key.split("/")
            current = root

            for part in parts[:-1]:
                current = current.setdefault(part, {})

            current[parts[-1]] = value

        return root

    def on_before_optimizer_step(self, optimizer: Optimizer) -> None:
        if self.use_moco:
            self.moco_momentum_update()

    def state_update(self) -> Literal['bidirectional', 'causal']:
        # Randomly draw the modality we want to train on (For the time relations)
        causal_p = self.p_causal_schedule()
        u = torch.rand((), generator=self.time_mask_switch_generator, device=self.device)
        mode: Literal['bidirectional', 'causal'] = "causal" if u < causal_p else "bidirectional"
        self.student.set_attention_modality(TimeMaskSwitchableProperties(mode=mode))
        if self.use_moco:
            self.momentum_student.set_attention_modality(TimeMaskSwitchableProperties(mode=mode))

        return mode

    # todo decompose per fare funzionare moco style
    def training_step(self, batch, batch_idx):
        mode = self.state_update()
        # Convert the batch to fp16 from quantized
        batch = self.dequantize(self.nest(batch), torch.float16)

        stud_out: WeaklySupervisedEegBaseModelOutputs = self.student(batch["student"], use_kd=True)
        self.momentum_modality_pooled = None

        if self.use_moco:
            with torch.no_grad():
                momentum_out: WeaklySupervisedEegBaseModelOutputs = self.momentum_student(batch["student"], use_kd=True)
                self.momentum_modality_pooled = {}

                for key, mv in momentum_out.multimodal_outs.items():
                    valid = self._get_y_valid(mv)
                    B, D = mv["data"].size(0), mv["data"].size(-1)
                    pooled_all = torch.zeros((B, D), device=mv["data"].device, dtype=mv["data"].dtype)

                    if valid.any():
                        pooled_valid = self._y_mean(mv, valid)
                        pooled_all[valid] = pooled_valid

                    self.momentum_modality_pooled[key] = pooled_all  # [B, D]

        with torch.inference_mode():
            teacher_out: MaskedContrastiveModelOutputs = self.teacher(batch["teacher"])

        return self._compute_step_metrics(stud_out, teacher_out, batch, 'train', mode)

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        out = {}
        mode: Literal['causal', 'bidirectional']
        # Convert the batch to fp16 from quantized
        batch = self.dequantize(self.nest(batch), torch.float16)

        with torch.inference_mode():
            teacher_out: MaskedContrastiveModelOutputs = self.teacher(batch["teacher"])

        for mode in ("causal", "bidirectional"):
            self.student.set_attention_modality(TimeMaskSwitchableProperties(mode=mode))
            stud_out: WeaklySupervisedEegBaseModelOutputs = self.student(batch["student"], use_kd=True)
            out[mode] = self._compute_step_metrics(stud_out, teacher_out, batch, 'val', mode)

        return out

    def on_train_batch_end(self, outputs: dict, batch: Any, batch_idx: int) -> None:
        # Every 10 batches we run the batch end operations
        if not batch_idx % 10 == 0:
            return

        _ = outputs.pop("loss")  # We have to ignore it
        if self.FUSED_KEY in outputs:
            fused_z = outputs.pop(self.FUSED_KEY)
            pivot_z = self._y_mean(outputs[self.PIVOT_KEY], self._get_y_valid(outputs[self.PIVOT_KEY]))
            # Compute and log the metrics.
            self._compute_batch_metrics(fused_z, pivot_z, outputs, 'train')

    def on_validation_batch_end(self, outputs: dict, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        # Every 10 batches we run the batch end operations
        if not batch_idx % 10 == 0:
            return  # todo vedi se si puo fare

        for mode, val in outputs.items():
            _ = val.pop("loss")  # We have to ignore it
            if self.FUSED_KEY in val:
                fused_z = val.pop(self.FUSED_KEY)
                pivot_z = self._y_mean(val[self.PIVOT_KEY], self._get_y_valid(val[self.PIVOT_KEY]))
                # Compute and log the metrics.
                self._compute_batch_metrics(fused_z, pivot_z, val, 'val', mode=mode)

    @staticmethod
    @torch.no_grad()
    def siglip_random_baseline(loss_fn, a, b):
        # shuffle targets to break alignment
        idx = torch.randperm(b.shape[0], device=b.device)
        return loss_fn(a, b[idx])

    def compute_kd_loss(self, student_out: dict[str, MaskedValue], teacher_out: MaskedContrastiveModelOutputs,
                        step_type: Literal['train', 'val', 'test']) -> torch.Tensor:
        loss = .0
        n = 0
        for key in teacher_out.keys():
            if key not in student_out:
                continue  # This element is not KD or is absent from teacher so we cannot learn from it
            student_data, teacher_data = student_out[key]["data"], teacher_out[key]['data']
            modality_loss = self.kd_losses[key](student_data, teacher_data)
            rand_baseline = self.siglip_random_baseline(self.kd_losses[key], student_data, teacher_data, )
            self.log(f"{step_type}/kd/{key}/rand", rand_baseline, on_epoch=True, on_step=True, prog_bar=False)
            self.log(f"{step_type}/kd/{key}/loss", modality_loss, on_epoch=True, on_step=True, prog_bar=False)
            loss = loss + modality_loss
            n += 1

        # Normalize so that missing modalities don't spike up the loss
        loss = loss / max(1, n)

        self.log(f"{step_type}/kd/loss", loss, on_epoch=False, on_step=True, prog_bar=True)
        self.log(f"{step_type}/kd/n_modalities", float(n), on_epoch=True, on_step=True, prog_bar=False)
        return loss

    @staticmethod
    def _get_y_valid(y: MaskedValue) -> torch.Tensor:
        return y["mask"].sum(dim=1) > 0

    # todo sistema. Prendi quello che sta facendo il modello (se fa CLS fai MAX?)

    def _y_mean(self, y: MaskedValue, valid_rows: torch.Tensor) -> torch.Tensor:
        y_before, mask = y["data"][valid_rows], y["mask"][valid_rows]
        pooling = self.student.pooling
        if isinstance(pooling, ClsPooling):
            pooling = MaskedAvgPooling()

        with torch.no_grad():
            return pooling(y_before, mask)

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
            zb_neg = None  # Negatives to add to batch
            if self.use_moco and step_type == "train":
                self.moco_init_queue(key, dim=z_pos.size(-1), device=z_pos.device)
                zb_neg = self.moco_queue[key]

            count_present += 1
            mod_loss = self.siglip_losses[key](fused_output[valid_rows], z_pos, zb_neg)

            if self.use_moco and step_type == "train":
                k = None if self.momentum_modality_pooled is None else self.momentum_modality_pooled.get(key)
                if k is not None:
                    self.moco_enqueue(key, k[valid_rows])  # optionally enqueue only rows matching valid_rows

            self.log(f"{step_type}/fusion/{mode}/{key}", mod_loss, on_epoch=True, on_step=True, prog_bar=False)
            self.log(f"{step_type}/fusion/{key}", mod_loss, on_epoch=True, on_step=True, prog_bar=False)
            base_loss = base_loss + mod_loss

        return base_loss / count_present

    @staticmethod
    def _top_1(x: torch.Tensor, y: torch.Tensor) -> Optional[torch.Tensor]:
        return EegAviKdVateMaskedSemiSupervisedModule._top_k(x, y, 1)

    @staticmethod
    def _top_k(x: torch.Tensor, y: torch.Tensor, k: int) -> Optional[torch.Tensor]:
        x = F.normalize(x, dim=-1)
        y = F.normalize(y, dim=-1)

        similarity = x @ y.T
        k = min(k, similarity.size(1))

        top_k = similarity.topk(k, dim=1).indices
        gt = torch.arange(similarity.size(0), device=similarity.device).unsqueeze(1)
        return (top_k == gt).any(dim=1).float().mean()
