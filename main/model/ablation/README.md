# Ablation studies

All ablations use the same data split, training budget, evaluation protocol, and seed set as the reference run. A single component is changed per run.

## MoCo

The momentum queue is disabled through the trainer configuration:

```yaml
use_moco: false
```

## Supporting modalities

Audio, text, and ECG are disabled separately. Video remains enabled as the primary supporting modality. Results include the absolute score and the change from the complete model.

## Knowledge distillation

The distillation objective is disabled through the model configuration:

```yaml
use_kd: false
```

## Interpretation

A consistent suboptimal configuration is acceptable for component comparison. Seed variance is estimated on the reference model first. Changes smaller than normal seed variation are considered inconclusive.
