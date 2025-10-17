from typing import Literal
import torch.nn.functional as F
import lightning as L
import torch.optim
from pytorch_lightning.utilities.types import OptimizerLRScheduler
from torchmetrics.functional import concordance_corrcoef

from main.model.EEGAVI.EEGAVI import WeaklySupervisedEEGAVIOutputs
from main.model.EEGAVI.base_model import WeaklySupervisedEegBaseModel, WeaklySupervisedEegBaseModelOutputs
from main.model.VATE.constrastive_model import MaskedContrastiveModel, MaskedContrastiveModelOutputs
from main.model.loss import masked_info_nce_2d, masked_cosine_similarity, SiglipLoss
from main.utils.data import MaskedValue


def sym_infonce(za: torch.Tensor, zb: torch.Tensor, tau=0.7):  # start higher tau
    za = F.normalize(za, dim=-1)
    zb = F.normalize(zb, dim=-1)
    # center to fight batch-wise shared direction
    za = F.normalize(za - za.mean(0, keepdim=True), dim=-1)
    zb = F.normalize(zb - zb.mean(0, keepdim=True), dim=-1)
    logits = (za @ zb.T).to(torch.float32) / tau
    t = torch.arange(za.size(0), device=za.device)
    return 0.5 * (F.cross_entropy(logits, t) + F.cross_entropy(logits.T, t))


class EegAviKdVateMaskedModule(L.LightningModule):
    pass


class EegAviKdVateMaskedSemiSupervisedModule(L.LightningModule):
    def __init__(self,
                 student: WeaklySupervisedEegBaseModel,
                 teacher: MaskedContrastiveModel,
                 kd_loss_weight: float,
                 fusion_loss_weight: float,
                 weakly_supervised_weight: float,
                 ecg_correction_weight: float,
                 lr: float = 1e-4,
                 kd_temperature: float = .2):
        super().__init__()
        self.student = student
        self.teacher = teacher
        # TODO Find defaults (questi sono per test a b=1)
        self.siglip_loss = SiglipLoss(init_tau=0.2, stop_grad_target=True)

        # Weight terms for loss parts
        self.alpha: float = kd_loss_weight
        self.beta: float = fusion_loss_weight
        self.gamma: float = weakly_supervised_weight
        self.theta: float = ecg_correction_weight
        self.kd_temperature = kd_temperature
        self.lr = lr

    def configure_optimizers(self) -> OptimizerLRScheduler:
        # todo warmup lr
        return torch.optim.Adam([
            {"params": self.student.parameters()},
            {"params": self.siglip_loss.parameters()}
        ], lr=self.lr, weight_decay=0
        )

    def compute_kd_loss(self, student_out: dict[str, MaskedValue], teacher_out: MaskedContrastiveModelOutputs) \
            -> float | torch.Tensor:
        numerator, denominator = .0, 0
        key: Literal['vid', 'txt', 'aud']
        for key in teacher_out.keys():
            if key not in student_out:
                continue  # Keys do never match so no kd on this.
            loss_k, n_rows = masked_info_nce_2d(
                za=student_out[key]["data"], za_mask=student_out[key]["mask"],
                zb=teacher_out[key]["data"], zb_mask=teacher_out[key]["mask"],
                tau=self.kd_temperature
            )

            loss_k /= torch.clamp(torch.log(torch.tensor(n_rows, device=loss_k.device, dtype=loss_k.dtype)), min=1e-6)
            # todo possibile problema potrebbe essere dovuto a batch size piccola:
            # Option 1: Gradient Accumulation (Recommended) Simulate larger batches without memory increase:
            #       Memory Bank alternativa piu robusta (Al momento siamo in un punto "good enough" per batch che abbiamo)
            #       Alternativa ancora sarebbe quella di usare sigliploss direttamente che dipende meno da bathc size?
            numerator += loss_k * n_rows  # Weighed by number of valid rows
            denominator += n_rows  # Track total number of valid rows

        loss = numerator / max(denominator, 1)
        self.log("kd_loss", loss, on_epoch=True)
        return loss * self.alpha

    def fusion_loss(self, after: torch.Tensor, before: MaskedValue) -> torch:
        # before is 3D. We mean over valid masked rows
        y_before = before["data"]
        mask_before = before["mask"].unsqueeze(-1)
        y_before = (y_before * mask_before).sum(dim=1) / mask_before.sum(dim=1)
        loss = sym_infonce(after, y_before, tau=0.3)
        return loss

    # todo sistemare
    def compute_fusion_loss(self, fused_output: torch.Tensor, modality_outputs: dict[str, MaskedValue]):
        device, dtype = fused_output.device, fused_output.dtype

        for key, value in modality_outputs.items():
            l = self.fusion_loss(fused_output, value)
            self.log("fus_" + key, l, on_epoch=True, prog_bar=True)

        num = torch.zeros((), device=device, dtype=dtype)
        den = torch.zeros((), device=device, dtype=dtype)

        # Build a union time mask across modalities to pool fused_output safely
        union_mask = None
        # TODO Qui ora mascherews sono diversa dimensione.
        for key, mv in modality_outputs.items():
            m = mv["mask"]
            if m is None:
                union_mask = None
                break
            if m.dim() > 2:
                m = m.any(dim=-1)  # [B,T]
            # union_mask = m if union_mask is None else (union_mask | m)
        fused_vec = fused_output
        # fused_vec = masked_mean_over_t(fused_output, union_mask)  # [B,D]

        # todo errore qui? -> Errore era numerico./
        # Resta nuovo errore: La diagonale e altri sample sono troppo simili.
        # TODO: Quale causa di questo? Troppo simili i sample?
        #       O il perceiver resmpler sbagliato?
        #       Todo forse attention pooling
        # Per-modality losses
        debug_pairs = []  # (za_valid, zb_valid) for debug only
        for key, value in modality_outputs.items():
            z, mask = value["data"], value["mask"]

            # Reduce patch axis properly
            if mask is not None and mask.dim() > 2:
                mask = mask.any(dim=-1)  # [B,T]
            # If z is [B,T,P,D], mean over patches; do masked mean if you have a patch mask per-timestep
            if z.dim() > 3:
                z = z.mean(dim=-2)  # [B,T,D]

            # Masked mean over time to [B,D]
            if mask is not None:
                w = mask.float().unsqueeze(-1)  # [B,T,1]
                z = (z * w).sum(dim=1) / w.sum(dim=1).clamp_min(1e-6)  # [B,D]
                valid_rows = mask.any(dim=1)  # [B]
            else:
                z = z.mean(dim=1) if z.dim() == 3 else z
                valid_rows = torch.ones(z.size(0), dtype=torch.bool, device=device)

            # Guard against degenerate rows
            z_norms = z.norm(dim=-1)
            valid_rows = valid_rows & (z_norms > 1e-6)

            if valid_rows.sum() == 0:
                continue

            za = fused_vec[valid_rows]
            zb = z[valid_rows].detach()

            # Symmetric InfoNCE (balanced); fixed tau for sanity
            def sym_infonce(za, zb, tau=0.7):  # start higher tau
                za = F.normalize(za, dim=-1)
                zb = F.normalize(zb, dim=-1)
                # center to fight batch-wise shared direction
                za = F.normalize(za - za.mean(0, keepdim=True), dim=-1)
                zb = F.normalize(zb - zb.mean(0, keepdim=True), dim=-1)
                logits = (za @ zb.T).to(torch.float32) / tau
                t = torch.arange(za.size(0), device=za.device)
                return 0.5 * (F.cross_entropy(logits, t) + F.cross_entropy(logits.T, t))

            assert not zb.requires_grad, "Target side is NOT detached"

            # keep for debug
            debug_pairs.append((za.detach(), F.normalize(z[valid_rows], dim=-1).detach()))
            if key == "eeg":
                continue

            loss_m = sym_infonce(za, zb, tau=0.3)

            fused_vec.retain_grad()
            (loss_m).backward(retain_graph=True)

            print("\nloss_m=", loss_m)
            n = valid_rows.sum()

            num = num + loss_m * n
            den = den + n  # <-- NOT n**2

            with torch.no_grad():
                za_n = F.normalize(za.to(torch.float32), dim=-1)
                zb_n = F.normalize(zb.to(torch.float32), dim=-1)
                S = za_n @ zb_n.T  # [n,n], now guaranteed in [-1, 1]
                diag = S.diag().mean().item()
                off = (S.sum() - S.diag().sum()) / (S.numel() - S.size(0))
                print(
                    f"diag={diag:.4f}, off={off:.4f}, gap={diag - off:.4f}, S_min={S.min().item():.4f}, S_max={S.max().item():.4f}")
                # sanity on norms
                print("||za|| mean:", za_n.norm(dim=-1).mean().item(), "||zb|| mean:", zb_n.norm(dim=-1).mean().item())

        # Optional: EEG–Audio geometry debug (only if you indeed have exactly 2 modalities here)
        if len(debug_pairs) >= 2:
            with torch.no_grad():
                # take first two for illustration
                za_eeg, z_eeg = debug_pairs[0]
                za_aud, z_aud = debug_pairs[1]
                # compute on the intersection of valid rows if you want stricter alignment; here we assume matched rows
                c = (z_eeg * z_aud).sum(dim=-1)
                bound = ((1 + c).clamp_min(0) / 2).sqrt()
                # fused cosines (use the same valid rows subset for each)
                cos_f_eeg = (F.normalize(fused_vec, dim=-1)[: z_eeg.size(0)] * z_eeg).sum(
                    dim=-1)  # simplistic alignment
                cos_f_aud = (F.normalize(fused_vec, dim=-1)[: z_aud.size(0)] * z_aud).sum(dim=-1)
                print("mean c:", float(c.mean()))
                print("mean bound:", float(bound.mean()))
                print("mean cos(f,EEG):", float(cos_f_eeg.mean()),
                      "\nmean cos(f,Audio):", float(cos_f_aud.mean()))

        loss = num / den.clamp_min(1e-6)
        self.log("unweighted_fusion_loss", loss, on_epoch=True)
        return loss * self.beta

    def a_compute_fusion_loss(self, fused_output: torch.Tensor, modality_outputs: dict[str, MaskedValue]):
        numerator = torch.zeros((), device=fused_output.device, dtype=fused_output.dtype)
        denominator = torch.zeros((), device=fused_output.device, dtype=fused_output.dtype)
        # TODO: Broken fusion module somehow.

        zs = []
        for key, value in modality_outputs.items():
            z, mask = value["data"], value["mask"]

            if mask.dim() == 3:
                mask = mask.any(dim=-1)
                # Usually stud_out_multimodal out: (b, T, P, D) (Differs for EEG)
            # Patch mean: (Has no mask patches are always max rank).
            if z.dim() > 3:
                # We have patch tokens so we mean over them
                z = z.mean(dim=-2)

            w = mask.float().sum(dim=-1, keepdim=True).clamp_min(1e-6)  # Normalization factor
            z = (z * mask.float().unsqueeze(-1)).sum(dim=-2) / w

            z_norms = z.norm(dim=-1)
            valid = (mask.any(dim=1)) & (z_norms > 1e-6)
            if valid.sum() == 0:
                continue

            n = valid.sum()
            zs.append(torch.nn.functional.normalize(z, dim=-1))
            if key == 'eeg':
                continue

            z = z.detach()
            modality_loss = self.siglip_loss(fused_output[valid], z[valid])

            numerator += modality_loss * n  #
            denominator += n ** 2  # Count the samples that have to do with loss calc

        c = (zs[0] * zs[1]).sum(-1)  # EEG–Audio agreement per sample
        bound = ((1 + c).clamp_min(0) / 2).sqrt()  # best achievable cosine to each (theory)
        fo = torch.nn.functional.normalize(fused_output, dim=-1)
        cos_f_eeg = (fo * zs[0]).sum(-1)
        cos_f_aud = (fo * zs[1]).sum(-1)

        print("mean c:", c.mean().item())
        print("mean bound:", bound.mean().item())
        print("mean cos(f,EEG):", cos_f_eeg.mean().item(), "\nmean cos(f,Audio):", cos_f_aud.mean().item())

        loss = numerator / denominator.clamp(min=1e-6)
        self.log("unweighted_fusion_loss", loss, on_epoch=True)
        return loss * self.beta

    def compute_supervised_loss(self, pred: torch.Tensor, target: torch.Tensor):
        # Compute concordance correlation coefficient that measures the agreement between two variables.
        # In emotion regression (valence, arousal, dominance), this is the standard metric and loss used in benchmarks. TODO verify
        # Correlation and agreement rather than absolute distance
        tol = 1e-8
        target_std = target.std(dim=0)
        ccc = concordance_corrcoef(pred[:, target_std > tol], target[:, target_std > tol]).mean()
        self.log("unweighted_supervised_loss", ccc, on_epoch=True)
        return (1 - ccc) * self.gamma

    def training_step(self, batch, batch_idx):
        stud_out: WeaklySupervisedEegBaseModelOutputs = self.student(batch["student"], use_kd=True)
        with torch.inference_mode():
            teacher_out: MaskedContrastiveModelOutputs = self.teacher(**batch["teacher"])

        loss = (
            # self.compute_kd_loss(student_out=stud_out.kd_outs, teacher_out=teacher_out)
            self.compute_fusion_loss(fused_output=stud_out.embeddings, modality_outputs=stud_out.multimodal_outs)
            # self.compute_supervised_loss(pred=stud_out.pred, target=batch["student"][Assessment.modality_code()])
        )

        # Optional term
        if False and "ecg" in stud_out.multimodal_outs:  # We have ECG
            loss = loss + self.theta * masked_cosine_similarity(
                za=stud_out.multimodal_outs["eeg"]["data"],
                # ECG goes through the Perceiver Resampler so it is 4D
                zb=stud_out.multimodal_outs["ecg"]["data"].mean(dim=-2),
                present=stud_out.multimodal_outs["eeg"]["mask"] & stud_out.multimodal_outs["ecg"]["mask"].any(dim=-1)
            )

        self.log("train_loss", loss, prog_bar=True, on_epoch=True)

        return loss
