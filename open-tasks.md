## Amigos

- Come fare con multi-person vids?
- Crop su volti e tratti come singole persone? Sembra opzione piú ragionevole
- Script decomposizione vid
- Video gruppi gestione
- Video Lunghi + segmenti di 20
- Ma fare estrazione testo solo 1 volta per tipo di video? I candidati non parlano dopotutto


## Loss

- studia bilanciamento

## Model

- Ripristino modality dropout
- Prova unimodale (skip layer?)
- Cambia parametrizzazioni
- Verifica che video senza pooling vada. Quanto è limite di size?
    - Meglio evitare: Memory explodes (non-solo ad inference che va ok ma during storage of pre-computed)

## Run

- preprocessing ds

## Manhob

- Usa manob per training
- preprocessing core
- pipeline implementaiton

## Deap

- Preprocessing pipeline fix

