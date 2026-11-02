# External model dependencies

This directory contains third-party model code and weights used by EEGAVI. These assets are not installed through `requirements.txt`.

## VATE

VATE is cloned into this directory:

```bash
git clone https://github.com/Ubriacopo/VATE.git VATE
```

The expected contrastive checkpoint location is `VATE/best_model_contrastive.pt`. \
Alternative locations are configured through `teacher_weights_path`.

## CBraMod

CBraMod is installed from the bundled package or an equivalent distribution. \
The preprocessing checkpoint is specified by `dataset.eeg_config.model_weights_path`.

## ECG-FM

The ECG pipeline expects a local feature extraction service.\
Service details are documented in [`ECG-FM/README.md`](ECG-FM/README.md). Its address is configured through `dataset.ecg_config.fm_endpoint`.

## Import resolution

Commands are run from the repository root. Unpackaged external projects are made available through `PYTHONPATH`:

```bash
export PYTHONPATH="$PWD/main/dependencies/VATE:$PYTHONPATH"
```

In PyCharm, `main/dependencies` is configured as a Sources Root. In VS Code, it is included in `python.analysis.extraPaths`.
