# Thesis notes

## Downstream protocol

- DREAMER excludes ECG. Its lead layout is not supported by the selected ECG foundation model.
- Fine-tuning uses three stages:
  1. Explore candidate settings.
  2. Train candidates for 15 epochs and retain the three best seeds.
  3. Fully train the four best runs and evaluate the strongest one on the test set.
