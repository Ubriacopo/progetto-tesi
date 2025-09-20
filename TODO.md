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


- Devo capire bene cosa volgio fare: FOGLIO E PENNA TI AIUTANO

- Vedi di fare questo se effettivamente teniamo all samples in un .pt anche di durata molto diversa
```py
with open("./resources/AMIGOS/processed-interleaved/P01_31.pt", "rb") as f_in, gzip.open("./resources/AMIGOS/processed-interleaved/P01_31.pt.gz", "wb") as f_out:
    shutil.copyfileobj(f_in, f_out)
```