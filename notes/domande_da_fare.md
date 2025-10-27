- When and where fusion (I guess I can only apply late fusion when I distil from VATE)
  https://medium.com/@raj.pulapakura/multimodal-models-and-fusion-a-complete-guide-225ca91f6861
  https://arxiv.org/html/2411.17040v1
  https://arxiv.org/pdf/2402.12030
- Fine tuning VIVT su AffectNet/FER?
- What is my goal?
- Also bones of posture fed to my model?
- When doing KD I feed an input x to two networks (teacher, student) When I do augmentations (like flipping images)
  should I do it for both? Or only for student?
- Ma dovrei secondo voi fare effettivo resampling dei frame in generale o per il mio modello lavorare su media.
    - VATE -> Sempre resampling (mi sembra corretto per vid),
    - FEEG -> Posso usare media (Un video point ha 32 frames ma posso avere n media?) Teoricamente la mia arch lo
      supporta
    - Comincia con semplice (DonwSampling) poi prova con complessi media (VIVIT non super adeguato ma forse altri nativ si)
      Altrimenti giro ViviT n volte e faccio poi quel mod
- Devo esplorare diverse possibili tecniche di distillazione?
- Modellino per ECG?
- Ma al posto di Vivit un bel clip-vit + custom stuff + PMA?
  - per-frame ViT (e.g., CLIP-ViT) → temporal mixer (GRU/TCN/S4) → compress (PMA/Perceiver) → Gated X-Attention.
  - Prova con TimesFormer che dovrebbe essere flessible per time sequences
- Ma devo fare più input stream video? Face + video che vedono?
- For what I could see the DREAMER dataset has no videos on the drive is it wanted?
- Late fusion per aux info che non hanno impatto temporale hanno senso/
- Ok so you say its better to ask myself: Is this data source modelled as a time sequence: If yes cat with xattn is a good pick If not it's better to just do late fusion (for example) with the output of my mlp and the z I get from xattn
  - context tokens or FiLM if the static info should steer attention -> lightweight): AV → KV, Tab → FiLM on Q (no tab tokens), head.
- Se introduci ECG: sincronizza con R-peaks (Pan-Tompkins) e crea heartbeat-centered windows; mappale sulle finestre EEG vicine.
- Audio paralinguistico (prosodia, pitch, energy) → a volte correla più con EEG/affetto del contenuto semantico.
- Use self-distillation over training iterations
- Projection to reconstruct e,v,a (all mods) -> self supervision. Siglip prima di gatedXAttention
- 
- Does it make sense for my foundation model to use the music track? It is kinda hard to get the audio and to map it to video but not impossible
- This supposes that I have the video I do not I only have face recordings and a highlight start index (in seconds) Most listed youtube links have been taken down So going for other clips of the same media might result in time unalignments
#### Seems like for DEAP the mapping just cant be done

- Consider our project. I have a limit of 8GB of VRAM which is tight. At the moment my video modality embeddings are of the shape [b, 1336, 768] These are of course very expensive. I thought it'd be better to not pass the [CLS] of the embedder (ViVit) but the patches for an early fusion. Should I give up to [CLS]? Is it enough for my EEG-FM?
- Keep [CLS]. 
Collapse patches → per-frame tokens (no params, no training).
Optionally temporal pool those per-frame tokens even more (still no training).
Use fp16, gradient checkpointing, and gradient accumulation.
If you really need a bottleneck, use a fixed (frozen) projection (Hadamard/orthogonal) so there’s nothing to train.
- 1) Deterministic pooling (no training)

- Assume ViVit gives you tokens laid out as [CLS] + (T × P) where P is #spatial patches per frame.
Spatial mean-pool to 1 token per frame:
```py
video_tokens: [B, 1 + T*P, 768]
cls = video_tokens[:, :1]
patches = video_tokens[:, 1:].view(B, T, P, 768)
frame_tok = patches.mean(dim=2)          # [B, T, 768]
compact = torch.cat([cls, frame_tok], 1) # [B, 1+T, 768]
```
That’s [B, 1+T, 768] instead of [B, 1336, 768].
If T=16, you drop from 1336 → 17 tokens per sample — a ~78× reduction in attention memory.

# video_tokens: [B, 1336, 768] = [CLS] + (T*P)
cls = video_tokens[:, :1]                     # [B, 1, 768]
patches = video_tokens[:, 1:]                 # [B, T*P, 768]
patches = patches.view(B, T, P, 768)          # know your T,P from ViVit config

frame_tok = patches.mean(dim=2)               # [B, T, 768]  (spatial mean per frame)
compact = torch.cat([cls, frame_tok], dim=1)  # [B, 1+T, 768]

# optional: temporal subsample to K tokens (no params)
K = 12
idx = torch.linspace(0, T-1, K, device=compact.device).round().long()
time_tok = compact[:, 1:][:, idx]             # [B, K, 768]
compact = torch.cat([cls, time_tok], dim=1)   # [B, 1+K, 768]


- Another thing If my modality drop a video the KD for video for that sample has to be discarded?
- Use VateVideoResamplerTransform for EEGAVI also?
- AL momento io fondo i channels e mi va bene ma sarebbe il caso di non farlo per EEG?
- LLaVA-style prefix alternativa (all togheter self attn)
- Dovrei passare a LaBram?