# EEGAVI

EEGAVI is a research prototype for affective EEG representation learning. It combines EEG with supporting video, audio, text, and ECG through gated cross-attention. Knowledge distillation guides the supporting modality adapters before fusion.

The project accompanies the thesis *A Multimodal Knowledge Distillation Model for Emotive EEG Characterization*.

## Why EEGAVI?

EEG datasets are expensive to collect. They require specialized equipment, controlled experiments, and participant time. Differences in electrode layouts and acquisition protocols also make datasets difficult to combine. These constraints limit the data available to train general EEG representations.

EEGAVI explores whether knowledge from other modalities can help address this limitation. It builds on frozen pretrained encoders and trains adapters and fusion layers for affective processing. The thesis investigates whether adding knowledge distillation improves the resulting representations over the CBraMod baseline.

## Why multimodal fusion?

Emotional states can appear in complementary signals: brain activity, facial expressions, speech, and physiological responses. Combining these views may reveal affective information that EEG alone does not capture reliably. Audio, video, and text also benefit from pretrained models developed on much larger corpora.

EEGAVI keeps EEG as the required pivot. Gated cross-attention lets its representations draw on the available supporting signals. This design aims to enrich EEG representations while accommodating datasets with different or missing modalities.

## Project status

The codebase supports dataset preprocessing, HDF5 sharding, pretraining, hyperparameter tuning, ablation studies, and downstream evaluation. \
It is an active research repository. Dataset access and pretrained weights are not included.

## Repository layout

- [`conf/`](conf/) contains Hydra configurations.
- [`main/core_data/`](main/core_data/) contains data structures, transforms, and loaders.
- [`main/dataset/`](main/dataset/) contains dataset adapters and preprocessing scripts.
- [`main/model/`](main/model/) contains EEGAVI, training code, and downstream tasks.
- [`report/tesi/`](report/tesi/) contains the thesis sources.
- [`notes/`](notes/) contains working research notes.
- [`notes/future-work.md`](notes/future-work.md) discusses limitations and possible improvements.
- [`results/best/`](results/best/) records the selected pretraining configuration.

## Setup

The project dependencies are installed in a dedicated Python environment:

```bash
python -m pip install -r requirements.txt
```

External model dependencies are described in [`main/dependencies/README.md`](main/dependencies/README.md). Machine-specific paths in the Hydra configuration require adjustment before experiments.

## Main workflows

All commands run from the repository root. Hydra overrides select datasets, pipelines, and experiments.

```bash
# Preprocess one dataset.
python -m main.dataset.script.preprocess dataset=amigos preprocessing=combined

# Build HDF5 shards.
python -m main.dataset.script.sharding sharding=amigos

# Train EEGAVI.
python -m main.model.script.train experiment=my-run
```

The default configurations contain local dataset and checkpoint paths. Overrides are supplied on the command line or through local configurations under `conf/local/`.

## Data

Pretraining uses AMIGOS, DEAP, EAV, and MAHNOB-HCI. The datasets have different modalities, channel layouts, sampling rates, and label spaces. Their adapters normalize these differences before sharding.

Access is subject to each dataset's license. Raw data must be obtained from the original providers.

## Thesis

The chapter map and build instructions are available in [`report/tesi/README.md`](report/tesi/README.md).
