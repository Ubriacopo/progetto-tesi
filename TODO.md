- Cambiare blocchi da 1s -> 4s (CBRaMod constraint)

> ### Per questo motivo devi cambiare il modello in: <br>
>  TODO si semplifica -> ogni mod uniformata a stesso timestep di EEG

> Valuta pos di feed forward in xattn


### KD


- Prova a fare plot di diagonale e somilgianze generali di un batch
- Costruisci un batch a mano chiaro (augmentation su dati)
- Problema forse è model bottleneck
- Forse resampler resta problema
- Altrimenti potrebbe essere proiezione in spazio di KD
- Quanto allineati dati

## MIGLIORIE
Tenere idea di contesto testo su itnervalli lunghi.
Ultimi 20secondi di paralto?


Sure — here it is **plain and concrete**, no jargon.

### What “train-set retrieval” means

Take a **training batch** (the same samples you trained on).

Pick **one modality as the query**, e.g. EEG.
Try to **retrieve the matching sample** in another modality, e.g. video.

If the model has overfit (or is fitting well), it should almost always pick the *correct* matching sample.

---

### Top-1 retrieval (the simplest check)

Example: **EEG → Video**

You have batch size = 12.

1. Take EEG embedding of sample *i*
2. Compute similarity to **all 12 video embeddings** in the batch
3. Sort them by similarity
4. Check:

   * Is the **correct video (same index i)** ranked **#1**?

If yes → success for that sample.

Top-1 accuracy =

> how many of the 12 queries retrieve the correct match at rank 1

---

### Top-k retrieval (a softer check)

Same as above, but:

* success if the correct match is in the **top k** (e.g. top-3)

Useful when supervision is noisy.

---

### Why this tells you about overfitting

* If **train-set top-1 ≈ 100%** but loss is still ~2.2
  → the model *has memorized the training set*, and the remaining loss is due to:

  * many negatives
  * temperature/logit scale limits
  * conflicting modality pairs

* If **train-set top-1 is low** (say 40–60%)
  → the model has **not fit the training data yet** (or something is wrong).

This is *much more informative* than the raw loss value.

---

### Do this for every modality pair

You said you train **all pairs**, so check all of them:

* EEG → Video
* Video → EEG
* EEG → Audio
* Audio → EEG
* EEG → Text
* Text → EEG
* … etc.

You’ll often see:

* strong pairs (e.g. vid↔aud) near 100%
* weak pairs (e.g. eeg↔txt) much lower

That tells you **which pair is setting the loss floor**.

---

### Minimal pseudocode (PyTorch-ish)

```python
# x_q, x_k: (B, D), already L2-normalized
sim = x_q @ x_k.T            # (B, B)
pred = sim.argmax(dim=1)     # best match
gt = torch.arange(B, device=sim.device)

top1_acc = (pred == gt).float().mean()
```

Top-k:

```python
topk = sim.topk(k=3, dim=1).indices
topk_acc = (topk == gt[:, None]).any(dim=1).float().mean()
```

---

### Bottom line

Loss = **how hard the objective is**
Retrieval = **whether the model actually learned the alignment**

For overfitting checks, **retrieval beats loss every time**.


Train-set retrieval (top-1 / top-k)
+
Validation retrieval
