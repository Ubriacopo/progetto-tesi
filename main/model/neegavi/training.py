import copy
import math
from typing import Optional, Literal

import torch
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch import nn
from torch.nn import functional as F
from transformers import get_cosine_schedule_with_warmup

from main.core_data.utils import unflatten, dequantize
from main.model.VATE.constrastive_model import MaskedContrastiveModel, MaskedContrastiveModelOutputs
from main.model.blocks.pooling import ClsPooling, MaskedAvgPooling
from main.model.blocks.time_masked import TimeMaskSwitchableProperties
from main.model.blocks.xattention import GatedXAttentionBlock
from main.model.loss import SiglipLoss, siglip_random_baseline
from main.model.neegavi.moco import MoCoAble
from main.model.neegavi.model import EegInterAviModel
from main.model.neegavi.train_utils import KdTrainDataModule
from main.model.neegavi.utils import EegBaseModelOutputs, WeaklySupervisedEegBaseModelOutputs, \
    retrieval_metrics_chunked
from main.utils.data import MaskedValue
from main.utils.logging import make_logger


class EasyEegAviKdVateMaskedModule(MoCoAble):
    def __init__(
            self,
            student: EegInterAviModel,
            teacher: MaskedContrastiveModel,
            datamodule: KdTrainDataModule,
            kd_loss_weight: float,
            attention_layers: int,
            fusion_loss_weight: float,
            lr: float,
            weight_decay: float = 0.01,
            max_warmup_steps: int = 1000,
            seed: int = 1,
            # MoCo relative parameters
            use_moco: bool = False,
            momentum: float = .995,
            queue_size: int = 1024,
            heavy_compute_interval: int = 10,
            batch_size=None,
            use_kd: bool = True,
            use_fusion: bool = True
    ):
        super().__init__(use_moco=use_moco, momentum=momentum, queue_size=queue_size)
        self.use_kd: bool = use_kd
        self.use_fusion: bool = use_fusion

        self.datamodule: KdTrainDataModule = datamodule
        self.siglip_losses: nn.ModuleDict = nn.ModuleDict()
        for fusion_metric in student.fusion_keys():
            loss_fn = SiglipLoss(init_tau=0.07, init_bias=-10, stop_grad_target=False)
            self.siglip_losses.add_module(fusion_metric, loss_fn)

        self.kd_losses: nn.ModuleDict = nn.ModuleDict()
        for kd_key in teacher.keys:
            loss_fn = SiglipLoss(init_tau=0.05, init_bias=-10, stop_grad_target=True)
            self.kd_losses.add_module(kd_key, loss_fn)

        self.time_mask_switch_generator = torch.Generator()
        self.time_mask_switch_generator.manual_seed(seed)
        self.inner_logger = make_logger(self.__class__.__name__)

        self.student: EegInterAviModel = student
        self.teacher: MaskedContrastiveModel = teacher

        self._validation_pairs = {}

        self.momentum_student: EegInterAviModel | None = None
        self.save_hyperparameters(ignore=[
            "datamodule", "student", "teacher", "fusion_metrics", "queue_ptr", "moco_queue", "momentum_out"
        ])

    def configure_optimizers(self) -> OptimizerLRScheduler:
        lr = self.hparams.lr

        decay_parameters = []
        no_decay_parameters = []
        for name, param in self.student.named_parameters():
            if not param.requires_grad:
                continue

            if (
                    param.ndim == 1 or
                    name.endswith(".bias") or
                    name.endswith("_gate") or
                    name.endswith(".gate") or
                    "embedding" in name.lower() or
                    "norm" in name.lower() or
                    "latents" in name.lower() or
                    "cls_token" in name.lower()
            ):
                no_decay_parameters.append(param)
            else:
                decay_parameters.append(param)
            # Verify counts

        self.inner_logger.info(f"\n=== Weight Decay Groups ===")
        self.inner_logger.info(f"WITH decay: {len(decay_parameters)} param groups")
        self.inner_logger.info(f"WITHOUT decay: {len(no_decay_parameters)} param groups")

        optimizer = torch.optim.AdamW(
            params=
            # Parameters from the Siglip losses for the fusion
            [{"params": i.parameters(), "lr": lr, "weight_decay": 0.0} for i in self.siglip_losses.values()]
            # Parameters of the Siglip losses for the kd
            + [{"params": i.parameters(), "lr": lr, "weight_decay": 0.0} for i in self.kd_losses.values()]
            # Model parameters
            + [{"params": decay_parameters, "lr": lr, "weight_decay": self.hparams.weight_decay}]
            + [{"params": no_decay_parameters, "lr": lr, "weight_decay": 0.0}],
            fused=True
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": dict(
                interval="step",
                frequency=1,
                # Scheduler for the learning rate
                scheduler=get_cosine_schedule_with_warmup(
                    optimizer=optimizer,
                    num_warmup_steps=min(self.hparams.max_warmup_steps,
                                         int(self.trainer.estimated_stepping_batches * 0.05)),  # 5 %
                    num_training_steps=self.trainer.estimated_stepping_batches,
                )
            )
        }

    def on_validation_epoch_start(self):
        self._validation_pairs = {}

    def on_train_start(self) -> None:
        # Generation of the time mask switching. It is just a training trick.
        self.time_mask_switch_generator = torch.Generator(device=self.device)
        self.time_mask_switch_generator.manual_seed(self.hparams.seed)

    def training_step(self, batch, batch_idx):
        batch = dequantize(self.datamodule.dequantize_keys(), unflatten(batch), torch.float16)
        # Randomly draw the modality we want to train on (For the time relations)
        causal_p = self.p_causal_schedule()
        u = torch.rand((), generator=self.time_mask_switch_generator, device=self.device)
        mode: Literal['bidirectional', 'causal'] = "causal" if u < causal_p else "bidirectional"
        self.student.set_attention_modality(TimeMaskSwitchableProperties(mode=mode))

        with torch.inference_mode():
            teacher_out: Optional[MaskedContrastiveModelOutputs] = None
            if self.use_kd:
                teacher_out = self.teacher(batch["teacher"])

            if self.use_moco and self.momentum_student is not None:
                self.momentum_student.set_attention_modality(TimeMaskSwitchableProperties(mode=mode))
                self.momentum_out = self.momentum_student(batch["student"], use_kd=self.use_kd)

        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("lr", lr, on_step=True, on_epoch=False, prog_bar=False)
        self.observe_xattn_gates()  # More metrics to see if there are problems here.
        stud_out: WeaklySupervisedEegBaseModelOutputs = self.student(batch["student"], use_kd=self.use_kd)
        return self.compute_step_metrics(stud_out, teacher_out, "train", batch_idx)["loss"]

    def validation_step(self, batch, batch_idx):
        mode: Literal['causal', 'bidirectional']
        batch = dequantize(self.datamodule.dequantize_keys(), unflatten(batch), torch.float16)

        with torch.inference_mode():
            teacher_out: Optional[MaskedContrastiveModelOutputs] = None
            if self.use_kd:
                # Calculate teacher embeddings only if we use them of course
                teacher_out = self.teacher(batch["teacher"])

        # We only evaluate on bidirectional from now on. It is just a learning trick in the architecture now.
        self.student.set_attention_modality(TimeMaskSwitchableProperties(mode="bidirectional"))
        stud_out: WeaklySupervisedEegBaseModelOutputs = self.student(batch["student"], use_kd=self.use_kd)
        loss_object = self.compute_step_metrics(stud_out, teacher_out, 'val', batch_idx)

        valid_p = stud_out.multimodal_outs[self.student.pivot.code]["mask"].sum(dim=1) > 0
        for k, mv in stud_out.multimodal_outs.items():
            valid_k = mv["mask"].sum(1) > 0
            valid_both = valid_p & valid_k

            if not valid_both.any():
                continue

            f = F.normalize(stud_out.cls[valid_both], dim=-1)
            e = F.normalize(self.y_mean(mv, valid_both), dim=-1)

            entry = self._validation_pairs.setdefault(k, {"f": [], "e": []})
            entry["f"].append(f)
            entry["e"].append(e)

        return loss_object["loss"]

    def compute_batch_metrics(self, step_type: Literal['train', 'val', 'test']):
        # Potential memory blow-up in on_validation_epoch_end.
        top_k = (1, 3, 5, 10)
        combined = '-'.join(map(str, top_k))
        mean_acc = {top: [] for top in top_k}
        mean_r_items, mrr_items = [], []

        for key, pe in self._validation_pairs.items():
            f = torch.cat(pe["f"], dim=0)
            e = torch.cat(pe["e"], dim=0)
            metrics = retrieval_metrics_chunked(f, e, chunk_size=256)
            mrr = metrics["mrr"]

            recalls = []
            for top in top_k:
                top_values = metrics["recalls"][top]
                recalls.append(top_values)

                self.log(f"{step_type}/fused/top{top}_{key}", top_values, on_epoch=True)
                mean_acc[top].append(top_values)

            # Mean Recall@K over selected Ks
            mean_r = metrics["mean_r"]
            mean_r_items.append(mean_r)

            mrr_items.append(mrr)

            self.log(f"{step_type}/fused/meanR@{combined}_{key}", mean_r, on_epoch=True)
            self.log(f"{step_type}/fused/mrr_{key}", mrr, on_epoch=True)

            alignment = metrics["alignment"]
            self.log(f"{step_type}/fused/alignment_{key}", alignment, on_epoch=True)

            # Margin between positives and typical negatives
            margin = metrics["margin"]
            self.log(f"{step_type}/fused/margin_{key}", margin, on_epoch=True)

        for top in top_k:
            if mean_acc[top]:
                top_mean_value = torch.stack(mean_acc[top]).mean()
                self.log(f"{step_type}/fused/top{top}_mean", top_mean_value, on_epoch=True)

        if mean_r_items:
            mean_r_mean = torch.stack(mean_r_items).mean()
            self.log(f"{step_type}/fused/meanR@{combined}_mean", mean_r_mean, on_epoch=True)

        if mrr_items:
            mrr_items_mean = torch.stack(mrr_items).mean()
            self.log(f"{step_type}/fused/mrr_mean", mrr_items_mean, on_epoch=True)

    def on_validation_epoch_end(self):
        self.compute_batch_metrics("val")

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def on_test_epoch_start(self) -> None:
        self._validation_pairs = {}

    def on_test_epoch_end(self) -> None:
        self.compute_batch_metrics("test")

    @torch.no_grad()
    def moco_momentum_update(self):
        m = self.momentum
        for model_parameter, momentum_parameter in zip(self.student.parameters(), self.momentum_student.parameters()):
            momentum_parameter.data.mul_(m).add_(model_parameter.data, alpha=1. - m)

    def on_fit_start(self) -> None:
        if self.use_moco:
            self.init_moco()

    def init_moco(self):
        self.momentum_student = copy.deepcopy(self.student)
        self.momentum_student.eval()
        for parameter in self.momentum_student.parameters():
            parameter.requires_grad_(False)

    def compute_kd_loss(self,
                        student_out: dict[str, MaskedValue],
                        teacher_out: MaskedContrastiveModelOutputs,
                        step_type: Literal['train', 'val', 'test']) -> float:
        loss, n = .0, 0
        is_train = step_type == "train"
        for key in teacher_out.keys():
            if key not in student_out:
                continue  # This element is not KD or is absent from teacher so we cannot learn from it
            student_data, teacher_data = student_out[key]["data"], teacher_out[key]['data']
            modality_loss = self.kd_losses[key](student_data, teacher_data)

            rand_baseline = siglip_random_baseline(self.kd_losses[key], student_data, teacher_data, )

            self.log(f"{step_type}/kd/{key}/rand", rand_baseline, on_epoch=True, on_step=is_train, prog_bar=False)
            self.log(f"{step_type}/kd/{key}/loss", modality_loss, on_epoch=True, on_step=is_train, prog_bar=False)

            # ---- KD diagnostic: positive-pair cosine similarity ----
            # Assumes last dim is embedding dim.
            s = F.normalize(student_data.reshape(-1, student_data.shape[-1]), dim=-1)
            t = F.normalize(teacher_data.reshape(-1, teacher_data.shape[-1]), dim=-1)
            pos_cos = (s * t).sum(dim=-1).mean()
            self.log(f"{step_type}/kd/{key}/cos", pos_cos, on_epoch=True, on_step=is_train, prog_bar=False)
            # Cosine gap against in-batch negatives
            sim = s @ t.T
            pos = sim.diag()
            if sim.shape[0] > 1:
                neg_mask = ~torch.eye(sim.shape[0], dtype=torch.bool, device=sim.device)
                neg_mean = sim[neg_mask].mean()

                cos_gap = pos.mean() - neg_mean
                self.log(f"{step_type}/kd/{key}/cos_gap", cos_gap, on_epoch=True, on_step=is_train, prog_bar=False)

            loss = loss + modality_loss
            n += 1

        # Normalize so that missing modalities don't spike up the loss
        loss = loss / max(1, n)
        self.log(f"{step_type}/kd/loss", loss, on_epoch=True, on_step=is_train, prog_bar=True)
        return loss

    def compute_fusion_loss(self,
                            fused_output: torch.Tensor,
                            modality_outputs: dict[str, MaskedValue],
                            step_type: Literal['train', 'val', 'test'], ) -> torch.Tensor:
        base_loss = torch.tensor(0.0, device=fused_output.device)
        count_present = 0

        for key, value in modality_outputs.items():
            valid_rows = value["mask"].sum(dim=1) > 0
            if not valid_rows.any():
                continue

            q = fused_output[valid_rows]
            # Negatives to add to batch
            # Always ensure queue exists (does not mean we use it)
            if self.use_moco and step_type == "train":
                self.moco_init_queue(key, dim=fused_output.size(-1), device=fused_output.device)

            use_moco = self.use_moco and step_type == "train" and self.momentum_out is not None and key in self.momentum_out.multimodal_outs
            zb_neg: Optional[torch.Tensor] = self.moco_queue[key] if use_moco else None

            if use_moco:
                mv_k = self.momentum_out.multimodal_outs[key]
                zb_pos = self.y_mean(mv_k, valid_rows).detach()  # [Nv, D]
            else:
                zb_pos = self.y_mean(value, valid_rows)

            count_present += 1
            mod_loss = self.siglip_losses[key](q, zb_pos, zb_neg)
            is_train = step_type == "train"
            self.log(f"{step_type}/fusion/{key}", mod_loss, on_epoch=True, on_step=is_train, prog_bar=False)

            base_loss = base_loss + mod_loss
            if use_moco:
                # Enqueue momentum positives
                self.moco_enqueue(key, zb_pos)

        return base_loss / max(1, count_present)

    def y_mean(self, y: MaskedValue, valid_rows: torch.Tensor):
        y_before, mask = y["data"][valid_rows], y["mask"][valid_rows]
        if isinstance(self.student.pooling, ClsPooling):
            return MaskedAvgPooling()(y_before, mask)

        return self.student.pooling(y_before, mask)

    def compute_step_metrics(
            self,
            stud_outs: EegBaseModelOutputs,
            teacher_outs: Optional[MaskedContrastiveModelOutputs],
            step_type: Literal['train', 'val', 'test'],
            batch_idx: int
    ) -> dict[str, torch.Tensor | MaskedValue]:
        loss = torch.tensor(0., device=stud_outs.embeddings['data'].device, dtype=stud_outs.cls.dtype)
        return_object: dict[str, torch.Tensor | MaskedValue] = dict(loss=loss)

        if self.use_kd and teacher_outs is not None:
            kd_loss = self.compute_kd_loss(student_out=stud_outs.kd_outs, teacher_out=teacher_outs, step_type=step_type)
            return_object["loss"] = return_object["loss"] + kd_loss * self.hparams.kd_loss_weight

        if self.use_fusion:
            is_train = step_type == "train"
            fusion_loss = self.compute_fusion_loss(stud_outs.cls, stud_outs.multimodal_outs, step_type)
            return_object["loss"] = return_object["loss"] + fusion_loss * self.hparams.fusion_loss_weight
            self.log(f"{step_type}/fusion-loss", fusion_loss, on_epoch=True, on_step=is_train, prog_bar=True)

        is_train = step_type == "train"
        self.log(f"{step_type}/loss", return_object["loss"], prog_bar=is_train, on_step=is_train, on_epoch=True)

        return return_object

    def p_causal_schedule(self, start: float = .05, end: float = .8, floor_bidirectional: float = .1) -> float:
        """
        Common practice: Current setups favors bidirectional at lower epochs and causal later ones.
        AntLM (2024): explicitly describes a unified framework that alternates/switches between causal LM (causal mask) and masked LM (bidirectional attention).
        :param start: The starting probability
        :param end: The target ending probability
        :param floor_bidirectional: Flooring to avoid one modality domination
        :return: The probability of being causal
        """
        t = min(self.trainer.global_step / self.trainer.max_steps, 1.0)
        # Cosine ramp
        p = start + 0.5 * (end - start) * (1 - math.cos(math.pi * t))
        return min(p, 1.0 - floor_bidirectional)  # At most p=0.9

    def observe_xattn_gates(self):
        xattn_layer: GatedXAttentionBlock
        for idx, xattn_layer in enumerate(self.student.gatedXAttn_layers):
            self.log(f"model/attn_gate_{idx}", xattn_layer.attn_gate, on_step=True, on_epoch=False, prog_bar=False)
            self.log(f"model/ff_gate_{idx}", xattn_layer.ff_gate, on_step=True, on_epoch=True, prog_bar=False)
