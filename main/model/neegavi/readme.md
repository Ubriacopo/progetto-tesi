# EEGAVI architecture

EEGAVI uses EEG as its pivot modality. Video, audio, text, and ECG provide supporting context. All foundation-model encoders are frozen. Training starts from their precomputed embeddings.

## Input shapes

| Modality | Embedding | Mask |
|---|---|---|
| EEG | `[B, T_e, C, D_e]` | `[B, T_e]` |
| Audio | `[B, T_a, P_a, D_a]` | `[B, T_a]` |
| Video | `[B, T_v, P_v, D_v]` | `[B, T_v]` |

`T` is the number of temporal steps. `P` is the number of tokens within a step. Text and ECG follow the same supporting-modality interface.

## Processing path

1. Perceiver resamplers compress the token axis of each supporting modality.
2. Modality adapters project all supports to a shared dimension.
3. Temporal positions and validity masks are expanded to match the compressed tokens.
4. Supporting tokens are concatenated as keys and values.
5. EEG tokens query them through gated cross-attention.
6. CLS representations provide the contrastive training targets.

Knowledge distillation is applied before fusion. It shapes the trainable adapters while the foundation encoders remain frozen.

## Temporal masking

Cross-attention receives EEG and support timestamps. The mask can be bidirectional, causal, or windowed. Padding masks are always applied. Causal runs must exclude support tokens from the future.
