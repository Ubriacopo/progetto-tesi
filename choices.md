- EEG_MODEL: CBraMod
- CLIP-loss -> No use SigLIP loss (better for use case)
  https://medium.com/@jiangmen28/siglip-vs-clip-the-sigmoid-advantage-457f1cb872ab

- Flamingo o Q‑Former / Query-Transformer (BLIP‑2 style) o Perceiver / Perceiver‑IO / Perceiver Resampler
- Resta sempre FLAMINGO! https://github.com/lucidrains/flamingo-pytorch/blob/main/flamingo_pytorch/flamingo_pytorch.py
- Quindi usiamo Flamingo Style FUSION + Gated cross‑attn / small fusion transformer outside an LLM

- Multi‑positive contrastive KD (simple & strong) or Distribution KD via similarity matching (CLIP‑style KD)
  multi‑positive InfoNCE. Alla fine usando siglip non dobbiamo fare questo su shared mod. Resta problema di EEG. 
> Downweighted on final loss to not destroy EEG deriving information

- BLIP‑2 Q‑Former (learned queries) vs FLAMINGO GxAttn (We can use both to see which best)

- Rotary Embeddings ?

> Is there a numerical difference if instead of BCE i do multi-class CE and pass video-text-audio all in the input.
> Do I gain in performance or is it better to keep it split? 


You’re right—I glossed over **multi-visual** packing. Here’s the precise, Flamingo-style picture, with EEG replacing text (queries) and **multiple videos** as media (keys/values).

# Shapes (with multi-visual support)

* **EEG (queries):**
  `eeg_tokens: (B, T_eeg, D)` — your EEG backbone outputs a *sequence* of tokens that will query the media. (This is the role text normally plays.)

* **Videos (media items):** input API takes **many media items**:
  `vision_x: (B, M, F, C, H, W)` where `M=#videos per sample`, `F=#frames per video`. This is exactly how OpenFlamingo expects multi-image/multi-video inputs. ([GitHub][1])

  After the vision encoder (pre-resampler), a video gives patchy features like you wrote:
  `(B, F, P, D_v)` (frames × patches), i.e. your `(B, T, P, D)`.

  Per **media item**, Flamingo applies a **Perceiver Resampler** → **fixed R tokens**:
  `video_latents: (B, M, R, D)` → flatten media axis →
  `media_kv: (B, M*R, D)`. This “fixed-R per item” is what lets Flamingo handle **arbitrarily interleaved multi-visual inputs**. ([arXiv][2])

* **Cross-attention (gated):**
  `Q = eeg_tokens (B, T_eeg, D)` attends to `K,V = media_kv (B, M*R, D)`. Flamingo’s *interleaving logic* maps query positions to the **right media item(s)** (originally via `<image>` / `<|endofchunk|>` markers in the text prompt). ([GitHub][1])

# How to wire the “multi-visual” mapping without text

You have two clean choices; both are faithful to Flamingo’s multi-visual design:

**A) Global attention to all videos**
All EEG tokens see all media tokens. Implementation: build a cross-attn mask that’s `True` for the whole `(M*R)` range at every EEG timestep.

**B) Per-video (or windowed) attention**
Map each EEG timestep (or chunk) to *one* (or a small set of) video item(s):

* Let `media_id_per_q: (B, T_eeg)` be integers in `[0, M-1]` (or a small list for k-window).
* Convert to a boolean mask `attn_mask: (B, T_eeg, M*R)` that selects exactly the `R` tokens of the chosen video(s). (This mirrors Flamingo’s `<image>` interleaving but with EEG-driven indices instead of text markers.) The OpenFlamingo API demonstrates the multi-media packing and the idea of aligning queries with per-item media chunks via special markers; here you’re providing that alignment directly via a mask. ([GitHub][1])

# Quick sanity checklist

* Input many videos: `vision_x (B, M, F, C, H, W)` ✔️ ([GitHub][1])
* Per video, encoder → resampler → `R` tokens: `(B, M, R, D)` → `(B, M*R, D)` ✔️ ([arXiv][2])
* EEG drives the model: `eeg_tokens (B, T_eeg, D)` as **queries** ✔️
* Provide an **explicit cross-attn mask** (global or per-item/window) so queries know *which* visual item(s) to use ✔️
* If you later reintroduce interleaving semantics, you can mimic Flamingo’s `<image>/<|endofchunk|>` behavior by segmenting EEG and switching the mask at boundaries (functionally identical to multi-visual text prompts). ([arXiv][2])

If you adopt option B (per-item/windowed attention), that’s a structural change—let’s remember to update your architecture diagram (`nonm-media-structure.png`) to show the **(B, M, R, D) → (B, M\*R, D)** packing and the **EEG→media** masking.

[1]: https://github.com/mlfoundations/open_flamingo "GitHub - mlfoundations/open_flamingo: An open-source framework for training large multimodal models."
[2]: https://arxiv.org/abs/2204.14198?utm_source=chatgpt.com "Flamingo: a Visual Language Model for Few-Shot Learning"
 
 

