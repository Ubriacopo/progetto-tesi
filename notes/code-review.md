# Code review and fixes

## Fixed defects

| Area | Defect | Correction |
|---|---|---|
| Causal attention | Future support tokens were visible; self-attention bypassed temporal restrictions. | Correct time comparison, masked self-attention, and propagation of attention mode to temporal encoders. |
| Attention scaling | Queries were scaled twice. | Scaling is delegated to PyTorch attention. |
| FP16 masked pooling | Denominator guards rounded to zero for entirely masked modalities, producing NaN outputs or gradients. | Representable positive denominators in KD and attention pooling. |
| Preprocessing | Segment cropping and shared media could mutate later segments. | Independent media copies and separate student/teacher inputs. |
| Labels | Floating-point rescaling was calculated but never assigned. | Rescaled values are stored without integer truncation. |
| FACED | Zero-based trial indices were shifted by one. | Direct lookup with range validation. |
| Raw EEG | Fine-tuning passed time-first tensors to CBraMod's channel-first interface. | Explicit input, mask, and output axis conversion. |
| Trial pooling | Padded windows influenced trial predictions. | Masked aggregation in mean-based probes. |
| Frozen backbones | Parent training calls re-enabled training mode. | Evaluation mode is preserved for frozen probes and teachers. |
| HDF5 | Rollover reused the previous file's capacity; partial batches were omitted from length. | Per-file capacity reset and ceiling batch counts. |
| Sampling | Block permutations ignored their local RNG; fractions selected whole shards. | Local deterministic permutations and bounded sample counts. |
| Dataset splits | The legacy sharder allocated windows instead of participants. | Participant-level splitting; missing participant metadata raises an error. |
| Regression | Singleton dimensions were removed; CLARE tracked three outputs instead of one. | Stable batch/output shapes and one-dimensional CLARE metrics. |
| CLARE classification | F1 was minimized; probe optimizers assumed fine-tuning interfaces. | F1 maximization and compatible probe optimizer paths. |
| Training | The data-module logger referenced a nonexistent attribute; test losses used validation names. | Correct logger initialization and test-specific loss logging. |
| Optimizers | Some trainable backbone or hidden-head parameters were omitted. | Inclusion of the relevant trainable parameters. |
| Utilities | Optional masks, post-KD adapters, dropout recovery, and transform helpers could raise exceptions. | Correct argument handling, mask alignment, and fallback construction. |
| Entry points | The tuning configuration name and legacy FACED trainer import were invalid. | Existing configuration and compatible trainer export. |
| Notebooks | A notebook was embedded as raw JSON; exploration setup was marked as a raw cell; gains referenced an undefined baseline. | Restored structure and executable setup; unsupported gain calculations were removed. |
| Persistence | Numeric experiment IDs changed type after CSV reload. | Stable string identifiers and parent-directory creation. |

## Validation

- Static parsing of 223 project Python files.
- Checks for undefined names and missing local import symbols.
- CPU regression tests in [`tests/`](../tests/).
- An isolated copy of the original code was tested to confirm regression failures.
- Initial regression suite: 43 tests pass after correction; 37 fail against the original code. Two additional tests cover FP16 masked-pooling outputs and gradients.
- No full preprocessing, GPU training, or checkpoint reproduction was performed.

The local test environment uses Python 3.14 and PyTorch 2.9.1. The complete pinned environment could not be installed because its SciPy build failed under Python 3.14. Existing local packages were reused for the CPU checks. Compatibility with the original training environment remains to be verified.

```bash
python -m pip install -r requirements-dev.txt
python -m pytest
```

The commands assume that the project runtime dependencies are already installed.

## Existing experiment artifacts

- Previously generated EEG segments and rescaled labels require regeneration in a new output directory. Existing preprocessing specifications can otherwise cause them to be skipped.
- FACED scores and CLARE checkpoint selection require reevaluation.
- Attention, sampling, pooling, and raw-EEG fixes change experiment behavior. Earlier metrics remain historical results until corrected runs are completed.
- No dataset, checkpoint, or result artifact was overwritten during the review.

## Remaining research limitations

- Causal attention does not make precomputed features causal. Whole-window encoders and transcript context may already contain future information.
- Several experimental heads assume a fixed number of trial windows. Their configured input lengths still require dataset-level validation.
- Relative gains in the modality-ablation notebook still require validation results from matched baseline seeds. Raw ablation scores remain available.
- Dataset-specific indexing, media synchronization, external ECG service behavior, and GPU execution require the original resources for complete validation.

Research extensions are tracked in [`future-work.md`](future-work.md).
