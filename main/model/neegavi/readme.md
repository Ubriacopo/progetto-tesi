embedders are frozen. I get input embeddings already
pivot is EEG rest are supporting modalities.s

EEG shape: [b, Te, c, De]
EEG mask: [b, Te]

Aud shape: [b, Ta, Pa, Da]
Aud mask: [b, Ta]

Vid shape: [b, Tv, Pv, Dv]
Vid mask: [b, Tv]

Step 1 : Ridurre la dimensione di Pa e Pv per comprimere informazioni maggiormente.
(POOLING) & (All support to same M) (Attn pooling o altro)
Vid [b, Tv, Pv, Dv] -> [b, Tv, M, Dv] (M << Pv)
Aud [b, Ta, Pa, Da] -> [b, Ta, M, Da]

Step 2: Remap to same D (supports)
Vid [b, Tv, M, Dv] -> [b, Tv, M, D]
Aud [b, Ta, M, Da] -> [b, Ta, M, D]

(Ma prima step 1 o step 2?)

Step 3: Costruire t_vid, t_aud
t_vid = [b, T] -> Quindi se T = 11 e ogni T dura 3 s -> [0, 3, 6, 9, 12, 15 ...]
t_aud = [b, T] -> Quindi se T = 34 e ogni T dura 0.96s -> [0, 0.96, 1.92 ...]

Step 4: Flatten M & T per ogni modality. Quindi cambiare anche la sua maschera e t_object
Vid [b, Tv, M, D] -> [b, Tv*M, D]
t_vid [b, Tv] -> [b, Tv*M]
vid_mask [b, Tv] -> [b, Tv*M]
Esempio se t_mask= [1,1,0...] (solo primi 6 secondi validi) e M= 2 allora:
vid_mask = [1,1,1,1,0...0]

A questo punto posso costruire kv come:
kv = torch.cat([Vid, Aud]
t_kv = torch.cat([t_vid, t_aud])
mask_kv= torch.cat([vid_mask, aud_mask])

Ora posso fare xattn con q=EEG, kv=kv
Devo passare a xattn anche mask_kv, mask_EEG, e t_kv e t_eeg
Con questi dati posso assicurami che (ad esempio)

- Se EEG valido a tempo fino al T=4 (4 secondi) (mask=[1,1,1,1,0....] prima di espandere su M)
    Devo fare attn su t_vid su tutto t_vid fino a 4 (primi 6 secondi visto che 4 >3 e devo prendere due step)
    Allora posso fare attn su t_aud quando t_aud > 3 e t_aud <4 (qui circa)

    Dovrei anche vedere cose del passato?
    Essendo un EEG ha senso che quello che provo deriva da tempi passati?
    Non diventa troppo il contesto? Potrei usare una window...
    Ho forse mal compreso come interagiscono i dati.

Attn part: