- Drop Text
    - Non sembra avere valore aggiunto
- Experiment with time series
- Suppose we want 32 frames per clip
```py
from transformers import VivitFeatureExtractor, VivitModel

feature_extractor = VivitFeatureExtractor(size=224, num_frames=32)
# MA NON ESISTE DANNATO!
```
- Configura viviprocessor
