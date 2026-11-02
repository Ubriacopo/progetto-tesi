# Design choices

- **EEG encoder:** CBraMod.
- **Contrastive loss:** SigLIP-style sigmoid loss. It avoids the global softmax used by CLIP and fits the available batch sizes.
- **Fusion:** Flamingo-style gated cross-attention outside the frozen encoders.
- **Token compression:** Perceiver resamplers for high-cardinality support tokens.
- **Distillation:** Cross-modal knowledge distillation before fusion. Its weight is limited to preserve EEG-specific information.
- **Training:** Frozen foundation encoders with trainable adapters and fusion layers.

Open comparisons include a BLIP-2-style Q-Former and rotary temporal embeddings.

Useful implementation reference: [flamingo-pytorch](https://github.com/lucidrains/flamingo-pytorch).
