For ablations, we have:

### MoCo-less (20H)

Simple change of config in `trainer/default.yaml`:

``` yaml
use_moco:  False
```

### Modalities-less (3 * 20H)

Here we go one modality at a time with exception made for Video that we consider core.<br>
So we try:

- Audio-less
- Text-less
- ECG-less

### Knowledge Distillation less (20H)

KD less is achieved by passing to the model the `use_kd` flag.
Just as MoCo less it is to be disabled in config ``:

``` yaml
use_kd:  False
```

## NOTA

> An ablation on a suboptimal but consistent config is still methodologically correct.

We can use this to do ablation also locally on smaller batch size (supposing optimal b!=32)

> Heuristic: require the ablation delta to be at least 2× the usual seed fluctuation

Seed noise I get from my first train If I do 3 seeds and fluctuation is 0.01 that is my order of reference for ablation difference