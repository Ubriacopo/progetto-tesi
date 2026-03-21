Best model configuration:

- **lr**: 3⁻⁴
- **batch_size**: 32
    - This one gotta be fixed if we ablate MoCo in hp search process because
      of machine limitations and accumulation not working as intended.
- **alpha**: 0.5
    - Weight of the KD
- **attn_layers**: 6
    - Number of attn layers
