from typing import Literal, Any, Optional

import lightning as L
import torch
import torch.nn.functional as F
from lightning.pytorch.utilities.types import OptimizerLRScheduler, STEP_OUTPUT
from torch import nn
from torchmetrics.functional import pearson_corrcoef, concordance_corrcoef

from main.core_data.media.assessment.assessment import Assessment
from main.model.VATE.constrastive_model import MaskedContrastiveModel, MaskedContrastiveModelOutputs
from main.model.loss import SiglipLoss
from main.model.neegavi.blocks import TimeMaskSwitchableProperties
from main.model.neegavi.model import EegInterAviModel
from main.model.neegavi.pooling import ClsPooling, MaskedAvgPooling
from main.model.neegavi.utils import WeaklySupervisedEegBaseModelOutputs, EegBaseModelOutputs
from main.model.neegavi.xattention import GatedXAttentionBlock
from main.utils.data import MaskedValue
from main.utils.logging import make_logger


class EegAviKdVateMaskedSemiSupervisedModule(L.LightningModule):
    FUSED_KEY: str = "fused"
    PIVOT_KEY: str = 'eeg'

    def __init__(
            self,
            student: EegInterAviModel, teacher: MaskedContrastiveModel,
            dequantize_keys: list[str],
            kd_loss_weight: float, fusion_loss_weight: float, weakly_supervised_weight: float,
            fusion_metrics: list[str], kd_keys: list[str], lr: float, kd_temperature: float,
            bidirectional_p: float = .9,  # For ATTN
            seed: int = 1, batch_size=None
    ):
        super().__init__()
        self.batch_size = batch_size

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
        self.use_supervised_loss = True

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
        self.gamma: float = weakly_supervised_weight

        self.k: int = 5

        # Utils
        self._n_causal: int = 0
        self._n_bidirectional: int = 0

    # todo ramp up per other losses than supervised? Or just use supervised as aux
    def configure_optimizers(self) -> OptimizerLRScheduler:
        params = []

        params += [
            # siglip_common_optim_configs for Fusion
            {"params": i.parameters(), "lr": self.lr * 10, "weight_decay": 0.0}
            for i in self.siglip_losses.values()
        ]

        params += [
            # siglip_common_optim_configs for KD
            {"params": i.parameters(), "lr": self.lr * 10, "weight_decay": 0.0}
            for i in self.kd_losses.values()
        ]

        params += [{"params": self.student.parameters(), "lr": self.lr}]  # Student parameters
        return torch.optim.Adam(weight_decay=.01, params=params, fused=True)

    def _compute_step_metrics(self, stud: EegBaseModelOutputs,
                              teacher: MaskedContrastiveModelOutputs,
                              batch, step_type: Literal['train', 'val', 'test'],
                              mode: Literal['bidirectional', 'causal']):
        return_object: dict[str, torch.Tensor | MaskedValue] = dict(
            loss=torch.tensor(0, device=stud.embeddings['data'].device))
        if self.use_kd_loss:
            kd_loss = self.compute_kd_loss(student_out=stud.kd_outs, teacher_out=teacher, step_type=step_type)
            return_object["loss"] = return_object["loss"] + kd_loss * self.alpha

        if self.use_fusion_loss:
            fusion_loss = self.compute_fusion_loss(
                fused_output=stud.cls, modality_outputs=stud.multimodal_outs, step_type=step_type, mode=mode
            )
            return_object["loss"] = return_object["loss"] + fusion_loss * self.beta
            # For later evaluations

            return_object[self.FUSED_KEY] = stud.cls.detach()
            for key, masked_value in stud.multimodal_outs.items():
                masked_value["data"] = masked_value["data"].detach()
                masked_value["mask"] = masked_value["mask"].detach()
                return_object[key] = masked_value

        self.log(f"{step_type}/loss-{mode}", return_object["loss"], prog_bar=True, on_step=True, on_epoch=True)
        return return_object

    # todo expensive only on some batches
    def _compute_batch_metrics(self, fused_z, pivot_z, outputs: dict, step_type: Literal['train', 'val', 'test'],
                               mode: Optional[Literal['bidirectional', 'causal']] = None):
        # Euclidean distance between the two embedding spaces
        dist = (pivot_z - fused_z).norm(dim=1).mean()
        self.log(f"{step_type}/norm(eeg-fused)", dist, on_step=False, on_epoch=True, prog_bar=True)
        mode_prefix = "" if mode is None else f"{mode}/"

        for key, embedding in outputs.items():
            valid = self._get_y_valid(embedding)
            if not valid.any():
                continue  # This modality cannot be evaluated

            embedding = self._y_mean(embedding, valid)
            # TOP-1 FUSED
            t1_fused = self._top_1(fused_z[valid], embedding)
            t3_fused = self._top_k(fused_z[valid], embedding, 3)
            tk_fused = self._top_k(fused_z[valid], embedding, self.k)
            t1_fused_rev = self._top_1(embedding, fused_z[valid])
            t3_fused_rev = self._top_k(embedding, fused_z[valid], 3)
            tk_fused_rev = self._top_k(embedding, fused_z[valid], self.k)

            self.log(f"{step_type}/{mode_prefix}fused/top1_{key}", t1_fused,
                     prog_bar=False, on_step=False, on_epoch=True)
            self.log(f"{step_type}/{mode_prefix}fused/top1_{key}_R", t1_fused_rev,
                     on_step=False, on_epoch=True)
            self.log(f"{step_type}/{mode_prefix}fused/top3_{key}", t3_fused,
                     prog_bar=False, on_step=False, on_epoch=True)
            self.log(f"{step_type}/{mode_prefix}fused/top3_{key}_R", t3_fused_rev,
                     on_step=False, on_epoch=True)
            self.log(f"{step_type}/{mode_prefix}fused/top{self.k}_{key}", tk_fused,
                     prog_bar=False, on_step=False, on_epoch=True)
            self.log(f"{step_type}/{mode_prefix}fused/top{self.k}_{key}_R", tk_fused_rev,
                     on_step=False, on_epoch=True)

            if key == self.PIVOT_KEY:
                continue

            # PIVOT
            t1_pivot = self._top_1(pivot_z[valid], embedding)
            t1_pivot_rev = self._top_1(embedding, pivot_z[valid])

            self.log(f"{step_type}/{mode_prefix}pivot/top1_{key}", t1_pivot,
                     prog_bar=False, on_step=False, on_epoch=True)
            self.log(f"{step_type}/{mode_prefix}pivot/top1_{key}_R", t1_pivot_rev,
                     on_step=False, on_epoch=True)

            delta = t1_fused - t1_pivot
            self.log(f"{step_type}/{mode_prefix}delta_{key}", delta,
                     prog_bar=False, on_step=False, on_epoch=True)

    warmup_threshold: float = .5
    causal_threshold: float = .8

    def p_causal_schedule(self):
        # AntLM (2024): explicitly describes a unified framework that alternates/switches between causal
        # LM (causal mask) and masked LM (bidirectional attention).
        # Current setups favors bidirectional at lower epochs and causal later ones
        # TOO WEAK
        # initial_causal = 1 - self.bidirectional_p
        # return min(initial_causal + self.bidirectional_p * self.current_epoch / self.trainer.max_epochs, 1.0)
        # TODO parametrizza
        frac = self.current_epoch / self.trainer.max_epochs
        if frac < self.causal_threshold:
            return .1
        elif frac < self.causal_threshold:
            return .4

        return .5

    def on_train_epoch_start(self) -> None:
        self._n_causal = 0
        self._n_bidirectional = 0

    def on_train_epoch_end(self) -> None:
        total = self._n_causal + self._n_bidirectional
        self.log("train/frac_causal", self._n_causal / max(1, total), on_epoch=True)
        self.log("train/frac_bidi", self._n_bidirectional / max(1, total), on_epoch=True)

    def on_train_start(self) -> None:
        self.time_mask_switch_generator = torch.Generator(device=self.device)
        self.time_mask_switch_generator.manual_seed(self.base_seed)

    def dequantize(self, batch: dict, dtype=torch.float16):
        # Student part:
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

    def training_step(self, batch, batch_idx):
        # Randomly draw the modality we want to train on (For the time relations)
        causal_p = self.p_causal_schedule()
        u = torch.rand((), generator=self.time_mask_switch_generator, device=self.device)
        mode: Literal['bidirectional', 'causal'] = "causal" if u < causal_p else "bidirectional"
        # Convert the batch to fp16 from quantized
        batch = self.dequantize(self.nest(batch), torch.float16)

        if mode == "bidirectional":
            self._n_bidirectional += 1
        else:
            self._n_causal += 1

        self.student.set_attention_modality(TimeMaskSwitchableProperties(mode=mode))

        stud_out: WeaklySupervisedEegBaseModelOutputs = self.student(batch["student"], use_kd=True)
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
        for key in teacher_out.keys():
            if key not in student_out:
                continue  # This element is not KD or is absent from teacher so we cannot learn from it
            student_data, teacher_data = student_out[key]["data"], teacher_out[key]['data']
            modality_loss = self.kd_losses[key](student_data, teacher_data)
            rand_baseline = self.siglip_random_baseline(self.kd_losses[key], student_data, teacher_data, )
            self.log(f"{step_type}/kd/{key}/rand", rand_baseline, on_epoch=True, on_step=False, prog_bar=True)
            self.log(f"{step_type}/kd/{key}/loss", modality_loss, on_epoch=True, on_step=False, prog_bar=True)
            loss = loss + modality_loss

        self.log(f"{step_type}/kd/loss", loss, on_epoch=True, on_step=False, prog_bar=True)
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

            count_present += 1
            mod_loss = self.siglip_losses[key](fused_output[valid_rows], self._y_mean(value, valid_rows))
            self.log(f"{step_type}/fusion/{mode}/{key}", mod_loss, on_epoch=True, on_step=True, prog_bar=True)
            base_loss = base_loss + mod_loss

        return base_loss / count_present

    def compute_supervised_loss(self, pred: torch.Tensor, target: MaskedValue,
                                step_type: Literal['train', 'val', 'test']) -> torch.Tensor:
        # Compute concordance correlation coefficient that measures the agreement between two variables.
        # In emotion regression (valence, arousal, dominance), this is the standard metric and loss used in benchmarks.
        #
        # Correlation and agreement rather than absolute distance.

        tol = 1e-8
        T = 2  # warmup length (epochs) – tune this

        mask = target["mask"].any(dim=-1)
        # Drop missing rows.
        y = target["data"][mask]
        pred = pred[mask]

        t_std = y.std(dim=0, unbiased=False)
        p_std = pred.std(dim=0, unbiased=False)
        std_mask = (t_std > tol) & (p_std > tol)

        if std_mask.any():
            # todo why never see this
            pred, target = pred[:, std_mask].float(), y[:, std_mask].float()
            pearson = pearson_corrcoef(pred, target).mean().float()
            concordance = concordance_corrcoef(pred, target).mean().float()
            w = min(1.0, float(self.current_epoch) / T)  # or cosine ramp
            one = pred.new_tensor(1.0)  # ensures dtype/device match

            loss = (1 - w) * (one - pearson) + w * (one - concordance)
            self.log(f"{step_type}/supervised (CCC & Pearson)", loss, on_epoch=True, on_step=True, prog_bar=True)
            loss = loss.to(pred.dtype)
            return loss

        elif mask.any():
            loss = F.mse_loss(pred, y).float()
            self.log(f"{step_type}/supervised", loss, on_epoch=True, on_step=True, prog_bar=True)
            return loss

        return torch.tensor(.0, device=pred.device, dtype=pred.dtype)

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

    def observe_xattn_gates(self):
        xattn_layer: GatedXAttentionBlock
        for idx, xattn_layer in enumerate(self.student.base_model.gatedXAttn_layers):
            self.log(f"model/attn_gate_{idx}", xattn_layer.attn_gate, on_step=False, on_epoch=True, prog_bar=True)
            self.log(f"model/ff_gate_{idx}", xattn_layer.ff_gate, on_step=False, on_epoch=True, prog_bar=True)
