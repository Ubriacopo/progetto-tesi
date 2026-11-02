# Experiment notes

## Dataset size

The combined corpus contains about 82,000 samples. This is large for EEG research, but modest for modern audio and video representation learning.

| Dataset | Approximate samples |
|---|---:|
| AMIGOS | 22,205 |
| DEAP | 11,731 |
| MAHNOB-HCI | 915 |
| EAV | 40,000 |

Counts are provisional. Regenerate them after the final preprocessing pass.

## Training

Datasets differ greatly in size. Weighted sampling with a soft proportional cap is proposed. This preserves coverage without allowing EAV to dominate each epoch.
