# Selected model

Three candidates reached full training. The first configuration achieved the strongest validation MRR.

| Learning rate | Batch | KD weight | Cross-attention layers | Peak MRR | Iteration |
|---:|---:|---:|---:|---:|---:|
| `3e-4` | 32 | 0.5 | 6 | **0.74** | 45 |
| `1e-3` | 64 | 1.0 | 4 | 0.64 | 46 |
| `3e-4` | 64 | 0.5 | 4 | 0.63 | 19 |

## Test retrieval metrics

| Modality | Margin | Mean R@1/3/5/10 | MRR |
|---|---:|---:|---:|
| Audio | 0.1720 | 0.9701 | 0.9510 |
| ECG | 0.1147 | 0.8473 | 0.7826 |
| EEG | 0.2538 | 0.9971 | 0.9956 |
| Text | 0.0927 | 0.3810 | 0.3023 |
| Video | 0.1597 | 0.7203 | 0.6117 |
| Mean | — | 0.7832 | 0.7287 |

Values are rounded to four decimal places. \
The original experiment artifacts are retained for exact values and reproducibility.
