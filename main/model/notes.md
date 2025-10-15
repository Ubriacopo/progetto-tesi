Yep — that’s a good way to think about it.
With Flamingo-style fusion (`fused = EEG + σ(gate) * CrossAttn(EEG ← KV)`), the query stream (EEG) already has a privileged residual path. If you optimize a single contrastive term “fused ↔ EEG & KV all together,” the easiest solution is to hug EEG and ignore KV. Splitting the objective fixes the incentive:

# What to optimize (two complementary terms)

1. **KV-matching (make fused actually use Audio/Video/Text)**
   Treat *all non-EEG* modalities as **multi-positives** for the fused vector. This pulls the fused embedding toward information present in KV.

2. **EEG-preserve (don’t destroy the original EEG semantics)**
   A gentle constraint so the residual identity doesn’t get obliterated, but small enough that KV can still move the needle.

Then combine with weights:
[
\mathcal{L} = \lambda_{\text{kv}},\mathcal{L}*{\text{kv}} ;+; \lambda*{\text{q}},\mathcal{L}_{\text{q}} ;+; \text{(KD / supervised, if used)}.
]

# Drop-in sketches

## 1) Multi-positive InfoNCE for KV

Use the **same rows** (intersection of valid samples) and project + L2-normalize.

```python
def masked_mean_over_p(x, m):  # [B,T,P,D],[B,T,P] -> [B,T,D]
    if m is None: return x.mean(2)
    w = m.float().unsqueeze(-1); return (x*w).sum(2) / w.sum(2).clamp_min(1e-6)

def masked_mean_over_t(x, m):  # [B,T,D],[B,T] -> [B,D]
    if m is None: return x.mean(1)
    w = m.float().unsqueeze(-1); return (x*w).sum(1) / w.sum(1).clamp_min(1e-6)

def multi_positive_infonce(zq, zk_list, tau=0.3):
    zq = F.normalize(zq, -1)
    Z  = [F.normalize(zk, -1) for zk in zk_list]
    logits_all = torch.cat([zq @ zk.T for zk in Z], dim=1) / tau  # [B, P*B]
    pos = torch.stack([(zq * zk).sum(-1) / tau for zk in Z], dim=0)  # [P,B]
    return (torch.logsumexp(logits_all, dim=1) - torch.logsumexp(pos, dim=0)).mean()
```

Wiring inside `compute_fusion_loss_kv`:

```python
# 1) pool fused to [B,D] using union time mask across KV
fused_vec = masked_mean_over_t(fused_tokens, union_kv_mask)  # build union as OR over KV masks

# 2) collect KV targets (each to [B,D], detach for sanity / stability)
kv_vecs = []
rows_accum = None
for name, mv in modality_outputs.items():
    if name == "EEG": continue
    z, m = mv["data"], mv["mask"]
    if z.dim() > 3:  # [B,T,P,D]
        if m is not None and m.dim() > 2: m = m.any(-1)
        z = masked_mean_over_p(z, mv["mask"])
    if m is not None and m.dim() > 2: m = m.any(-1)
    z_vec = masked_mean_over_t(z, m)
    valid = (z_vec.norm(dim=-1) > 1e-6)
    kv_vecs.append(z_vec[valid].detach())  # stop-grad target
    rows_accum = valid if rows_accum is None else (rows_accum & valid)

# 3) align rows across all KV and fused
rows = rows_accum
zq = proj_fused(fused_vec[rows])                 # learnable
zk_list = [proj_kv[name](vec) for name, vec in zip([n for n in modality_outputs if n!="EEG"], kv_vecs)]
L_kv = multi_positive_infonce(zq, zk_list, tau=0.3)
```

> Why multi-positive? It removes the “pick one modality and ignore the others” incentive. The fused vector gets credit for aligning with **any** KV positive.

## 2) EEG-preserve (small weight)

Simplest: cosine preservation (or InfoNCE) **with a small λ_q**. Keep targets detached here too to prevent the “both sides move” collapse during sanity runs.

```python
eeg = modality_outputs["EEG"]["data"]; me = modality_outputs["EEG"]["mask"]
if eeg.dim() > 3 and me is not None and me.dim() > 2: me = me.any(-1)
eeg_vec = masked_mean_over_t(eeg, me)
rows_q = (eeg_vec.norm(dim=-1) > 1e-6)
zq_q = F.normalize(proj_fused(fused_vec[rows_q]), -1)
ze_q = F.normalize(proj_eeg(eeg_vec[rows_q]).detach(), -1)

# cosine preserve (bounded, gentle)
L_q = (1.0 - (zq_q * ze_q).sum(-1)).mean()
# or symmetric InfoNCE with fixed tau if you prefer
# L_q = 0.5*(F.cross_entropy((zq_q @ ze_q.T)/0.3, torch.arange(zq_q.size(0), device=zq_q.device)) +
#            F.cross_entropy((ze_q @ zq_q.T)/0.3, torch.arange(zq_q.size(0), device=zq_q.device)))
```

## 3) Combine

Start with (\lambda_{\text{kv}}=1.0), (\lambda_{\text{q}}=0.2) (so KV matters more). Tune until both `cos(f,KV)` rise and `cos(f,EEG)` stays reasonable.

```python
total = λ_kv * L_kv + λ_q * L_q
```

# Extra knobs that help this setup

* **Residual re-balancing**: the EEG residual can drown ctx.

  ```python
  self.beta = nn.Parameter(torch.tensor(0.5))  # sigmoid ~ 0.62
  fused_tokens = 0.8*eeg_tokens + torch.sigmoid(self.beta) * LN(out(ctx))
  ```
* **Pre-LN + correct scaling** in x-attn (`/ sqrt(d_head)`), or the softmax saturates.
* **Better KV pooling**: raw patch mean is often weak; use a tiny per-timestep attn-pool.
* **Per-modality projectors**: `proj_kv[name]` and `proj_eeg` (LN → Linear → GELU → Linear → LN). They let the model raise cross-modal agreement in a shared D.
* **Diagnostics**: log `||ctx||` vs `||EEG||`, attention entropy, and per-modality loss. If `||ctx|| << ||EEG||`, residual is still too dominant.

# Sanity you should see (single batch)

* With multi-positive KV loss on, **`cos(f,Audio)` turns positive** quickly (few epochs).
* If you add more KVs (video/text), both cosines trend toward the **√((1+c)/2)** bound based on their mutual agreements.
* The EEG-preserve term stops fused from drifting totally away from EEG, but doesn’t pin it so hard that KV can’t help.

---

**TL;DR**: Yes — compute a **KV fusion loss** (multi-positive) that *excludes* the query stream from the positives, and add a **small EEG-preserve term** separately. That lines the objective up with your Flamingo-style architecture: KV must actually influence the fused vector, while EEG stays recognizable.

(And when you settle on this, remember your earlier note: update `nonm-media-structure.png` to include “per-mod projectors → gated x-attn (EEG←KV) → fused projector → multi-positive loss + small EEG-preserve”.)
