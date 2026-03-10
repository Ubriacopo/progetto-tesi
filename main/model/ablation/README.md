For ablations, we have:

### MoCo-less

Simple change of config in `trainer/default.yaml`:

``` yaml
use_moco:  False
```

### Modalities-less

Here we go one modality at a time with exception made for Video that we consider core.


### Knowledge Distillation less

KD less is achieved by passing to the model the `use_kd` flag.
Just as MoCo less it is to be disabled in config ``:

``` yaml
use_kd:  False
```
