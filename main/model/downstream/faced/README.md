# FACED downstream protocol

FACED labels apply to complete trials, while EEGAVI operates on shorter windows. The trial remains the unit of supervision.

1. Each trial is split into fixed-length windows.
2. Each window is encoded with a frozen backbone.
3. Window-level CLS tokens are aggregated.
4. A small predictor is trained on the trial representation.
5. EEGAVI is compared with a frozen CBraMod baseline under the same protocol.

Initial window sizes are 16 or 32 seconds. The final size depends on the trial-length distribution. An 8-second window is an alternative if longer windows underperform or exceed memory limits.

Participant splits precede windowing. This prevents windows from the same participant or trial from crossing data splits.
