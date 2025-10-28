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
import gzip
import shutil

with open("./resources/AMIGOS/processed-interleaved/P01_31.pt", "rb") as f_in, gzip.open("./resources/AMIGOS/processed-interleaved/P01_31.pt.gz", "wb") as f_out:
    shutil.copyfileobj(f_in, f_out)
```


- Passare da segmenti di 32s a 16s per motivi di spazio
- Cambiare blocchi da 1s -> 4s (CBRaMod constraint)





### KD
- Prova a fare plot di diagonale e somilgianze generali di un batch
- Costruisci un batch a mano chiaro (augmentation su dati)
- Problema forse è model bottleneck
- Forse resampler resta problema
- Altrimenti potrebbe essere proiezione in spazio di KD
- Quanto allineati dati

## MIGLIORIE
Tenere idea di contesto testo su itnervalli lunghi.
Ultimi 20secondi di paralto?