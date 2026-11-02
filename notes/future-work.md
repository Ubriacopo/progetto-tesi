# Limitations and future work

Proposed improvements from the thesis and development notes. \
Comparisons require consistent data splits and training budgets.

## Limited and heterogeneous data

- **Limitation:** Uneven dataset sizes, electrode layouts, and modality coverage. Small accessible MAHNOB-HCI subset.
- **Possible solution:** Broader data coverage, channel-mapping checks, and leave-one-dataset-out evaluation.

## Precomputed embeddings

- **Limitation:** Fixed encoder outputs prevent online raw-signal augmentation and encoder adaptation.
- **Possible solution:** Cached augmented variants or selective encoder fine-tuning, subject to available compute.

## Temporal compression and causality

- **Limitation:** Four-second support tokens may lose brief events. Encoded text can already contain future context.
- **Possible solution:** Adaptive temporal resolution and causal feature extraction. A [temporal video resampler](../main/model/neegavi/todo.md) is also proposed.

## Uneven audio and text quality

- **Limitation:** Participant speech and stimulus audio have different roles. Transcription quality varies. Text has the lowest recorded retrieval MRR.
- **Possible solution:** Separate audio streams, alternative encoders, transcript filtering, and a text-free baseline.

## Missing modalities

- **Limitation:** Masks alone do not establish robustness to missing supports. ECG lead compatibility restricts coverage.
- **Possible solution:** Modality dropout, evaluation across support combinations, and alternative ECG encoders.

## Memory and storage costs

- **Limitation:** Large video features and padding increase storage and memory use. Sharding limits random access.
- **Possible solution:** Length-aware batching, storage compression, smaller token budgets, and larger shuffle buffers.

## Training stability and loss balance

- **Limitation:** MoCo reduced stability without improving performance in the tested setup. A fixed KD weight may affect modalities differently.
- **Possible solution:** SigLIP-only reference runs, modality-specific loss weights, and repeated-seed comparisons.

## Scope of evaluation

- **Limitation:** Retrieval alignment alone does not establish affective prediction quality or EEG-only utility.
- **Possible solution:** Matched CBraMod comparisons on EEG-only, multimodal, and cross-dataset tasks. Trial-level aggregation for recording-level labels.

## Priority

1. Causal feature extraction.
2. Audio and text quality checks.
3. Repeated-seed downstream comparisons.
4. Architectural extensions after validation of the baseline.
