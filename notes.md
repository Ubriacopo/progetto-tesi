## Dataset

Final dataset size: 82023 samples. Kinda small. But
> By EEG research norms, yes—84k samples is large. <br>
> By modern deep-learning video/audio norms, it is not.

#### Samples count per dataset
- AMIGOS: 22205 ~ 22k
  - fp32 total size: 328.2 GB + 215MB
  - int8 + scales size: -
- DEAP: 11731 ~ 12k
 
- MAHNOB: 915 ~ 1k 
  - fp32 total size: 9.3 GB + 3.1 MB
  - int8 + scales size: 2.3GB  + 492KB
- EAV: 40000 ~ 40k


## Trainer
How many batches_per_epoch considering dataset sizes differ?
Proportional cap (gentle normalization):