EEG_MODEL: CBraMod
CLIP-loss

Flamingo o Q‑Former / Query-Transformer (BLIP‑2 style) o Perceiver / Perceiver‑IO / Perceiver Resampler
Resta sempre FLAMINGO! https://github.com/lucidrains/flamingo-pytorch/blob/main/flamingo_pytorch/flamingo_pytorch.py
Quindi usiamo Flamingo Style FUSION + Gated cross‑attn / small fusion transformer outside an LLM

Multi‑positive contrastive KD (simple & strong) or Distribution KD via similarity matching (CLIP‑style KD) 
multi‑positive InfoNCE
> Downweighted on final loss to not destroy EEG deriving information

BLIP‑2 Q‑Former (learned queries) vs FLAMINGO GxAttn (We can use both to see which best)