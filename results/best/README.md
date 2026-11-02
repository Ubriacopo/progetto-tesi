# Selected pretraining configuration

| Parameter | Value |
|---|---:|
| Learning rate | `3e-4` |
| Batch size | `32` |
| KD weight (`beta`) | `0.5` |
| Cross-attention layers | `6` |

The batch size is fixed for MoCo ablations. The local training setup cannot reproduce the original effective batch size through gradient accumulation.

Detailed retrieval metrics are recorded in [`main/model/neegavi/best_model.md`](../../main/model/neegavi/best_model.md).
