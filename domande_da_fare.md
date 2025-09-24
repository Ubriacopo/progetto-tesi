- Con distillation faccio anche per EEG? Self distillation o fine tuning di foundation qui?
- (Errore di VATE constrastive) (fa audio-audio in model si vede)
- Domanda VATE: Ma prende sempre e solamente i primi 32 frame senza esitare?
  Non fa down sampling dei frame prima magari? Non capisco dal codice ho guardato e riguardato
- Il mio foundation model ritorna embeddings per ogni modality come VATE? Yes cosi fanno i foundaiton model
  Guarda per reference: https://github.com/openai/CLIP/blob/main/clip/model.py
- When and where fusion (I guess I can only apply late fusion when I distil from VATE)
  https://medium.com/@raj.pulapakura/multimodal-models-and-fusion-a-complete-guide-225ca91f6861
  https://arxiv.org/html/2411.17040v1
  https://arxiv.org/pdf/2402.12030
- Fine tuning VIVT su AffectNet/FER?
- What is my goal?
- Also bones of posture fed to my model?
- When doing KD I feed an input x to two networks (teacher, student) When I do augmentations (like flipping images)
  should I do it for both? Or only for student?
- Ma dovrei secondo voi fare effettivo resampling dei frame in generale o per il mio modello lavorare su media.
    - VATE -> Sempre resampling (mi sembra corretto per vid),
    - FEEG -> Posso usare media (Un video point ha 32 frames ma posso avere n media?) Teoricamente la mia arch lo
      supporta
    - Comincia con semplice (DonwSampling) poi prova con complessi media (VIVIT non super adeguato ma forse altri nativ si)
      Altrimenti giro ViviT n volte e faccio poi quel mod
- Devo esplorare diverse possibili tecniche di distillazione?
- Modellino per ECG?
- Ma al posto di Vivit un bel clip-vit + custom stuff + PMA?
  - per-frame ViT (e.g., CLIP-ViT) → temporal mixer (GRU/TCN/S4) → compress (PMA/Perceiver) → Gated X-Attention.
  - Prova con TimesFormer che dovrebbe essere flessible per time sequences
- Ma devo fare più input stream video? Face + video che vedono?
- For what I could see the DREAMER dataset has no videos on the drive is it wanted?
- Late fusion per aux info che non hanno impatto temporale hanno senso/
- Ok so you say its better to ask myself: Is this data source modelled as a time sequence: If yes cat with xattn is a good pick If not it's better to just do late fusion (for example) with the output of my mlp and the z I get from xattn
  - context tokens or FiLM if the static info should steer attention -> lightweight): AV → KV, Tab → FiLM on Q (no tab tokens), head.
- Se introduci ECG: sincronizza con R-peaks (Pan-Tompkins) e crea heartbeat-centered windows; mappale sulle finestre EEG vicine.
- Audio paralinguistico (prosodia, pitch, energy) → a volte correla più con EEG/affetto del contenuto semantico.
- Use self-distillation over training iterations
- Projection to reconstruct e,v,a (all mods) -> self supervision. Siglip prima di gatedXAttention
- 
- Does it make sense for my foundation model to use the music track? It is kinda hard to get the audio and to map it to video but not impossible
- This supposes that I have the video I do not I only have face recordings and a highlight start index (in seconds) Most listed youtube links have been taken down So going for other clips of the same media might result in time unalignments
#### Seems like for DEAP the mapping just cant be done