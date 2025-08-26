EEG_MODEL: CBraMod
CLIP-loss -> Nah better for us SigLIP loss
https://medium.com/@jiangmen28/siglip-vs-clip-the-sigmoid-advantage-457f1cb872ab


Flamingo o Q‑Former / Query-Transformer (BLIP‑2 style) o Perceiver / Perceiver‑IO / Perceiver Resampler
Resta sempre FLAMINGO! https://github.com/lucidrains/flamingo-pytorch/blob/main/flamingo_pytorch/flamingo_pytorch.py

Quindi usiamo Flamingo Style FUSION + Gated cross‑attn / small fusion transformer outside an LLM

Multi‑positive contrastive KD (simple & strong) or Distribution KD via similarity matching (CLIP‑style KD) 
multi‑positive InfoNCE
> Downweighted on final loss to not destroy EEG deriving information

BLIP‑2 Q‑Former (learned queries) vs FLAMINGO GxAttn (We can use both to see which best)

Rotary Embeddings ?

> Is there a numerical difference if instead of BCE i do multi-class CE and pass video-text-audio all in the input. 
Do I gain in performance or is it better to keep it split?