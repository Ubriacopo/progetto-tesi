- Con distillation faccio anche per EEG? Self distillation o fine tuning di foundation qui?
- (Errore di VATE constrastive) (fa audio-audio in model si vede)
- Domanda VATE: Ma prende sempre e solamente i primi 32 frame senza esitare?
  Non fa downsampling dei frame prima magari? Non capisco dal codice ho guardato e riguardato
- > Forse grazie al punto di prima ho capito come avere piu media. \
  Certo posso fare downsampling da 30fps a 16 (per stare 2-4s) ma se non bastano ho più media per un solo oggetto! (
  Quindi [1,2,32,224,224] ? O comunque il modello si
  deve arrangiare e fare come deve gli split -> Multi media di cui si parlava per gatedXAttention))
- Il mio foundation model ritorna embeddings per ogni modality come VATE? Yes cosi fanno i foundaiton model
  Guarda per reference: https://github.com/openai/CLIP/blob/main/clip/model.py
- What are logits (non fare documentati)
- When and where fusion (I guess I can only apply late fusion when I distil from VATE)
  https://medium.com/@raj.pulapakura/multimodal-models-and-fusion-a-complete-guide-225ca91f6861
  https://arxiv.org/html/2411.17040v1
  https://arxiv.org/pdf/2402.12030
- Nella nostra modalita EEG e veramente la modalita più importante di dati?
  O sono equivlaenti tutti?
- Per il momento sto usando il vostro modello pre-trained. Va bene o dovrei allenarne un altro?
- Fine tuning VIVT su AffectNet/FER?
- What is my goal?
- Also bones of posture fed to my modeL?
- When doing KD I feed an input x to two networks (teacher, student) When I do augmentations (like flipping images)
  should I do it for both? Or only for student?
- Ma dovrei secondo voi fare effettivo resampling dei frame in generale o per il mio modello lavorare su media.
    - VATE -> Sempre resampling (mi sembra corretto per vid),
    - FEEG -> Posso usare media (Un video point ha 32 frames ma posso avere n media?) Teoricamente la mia arch lo
      supporta
    - Comincia con semplice (DonwSampling) poi prova con complessi media (VIVIT non super adeguato ma forse altri nativ si)
      Altrimenti giro ViviT n volte e faccio poi quel mod