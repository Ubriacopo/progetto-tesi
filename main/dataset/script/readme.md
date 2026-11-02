# Dataset scripts

Dataset scripts are run from the repository root and use Hydra configuration groups.

```bash
# Extract valid intervals.
python -m main.dataset.script.extract_intervals dataset=amigos

# Generate model-ready features.
python -m main.dataset.script.preprocess dataset=amigos preprocessing=combined

# Pack processed samples into HDF5 shards.
python -m main.dataset.script.sharding sharding=amigos
```

Machine-specific paths belong in local overrides. Shared configurations remain portable.
