## 1 - Choices
TBA.

## 2- Possible improvements:

#### Saving memory by compressing data after generation

This would actually lead to a real improvement in space thanks to the numerous padding we encounter.
(Most samples don't reach the longest one and are 0 padded and masked)

#### Robust Logging
TBA

#### Download resources like input feed video and audio
Can be found online with some work.


Ciao Francesco, ho fatto le sistemazioni che mi hai segnato (le ho lasciate per il momento così da poter tracciare le modifiche).
Manca un pezzo che saprò scrivere quando ho ben chiaro cosa fare con il dataset VR
Ho anche scritto il capitolo dello stato dell'arte
Quando cito delle fonti basta farlo la prima volta che ne parlo o sarebbe il caso di farlo ogni qual volta che passa un po' di tempo (ad esempio tra capitoli)?

Ho comunque condotto le quasi finito le ablation e sto preparando i downtream task
In ablation per ora ho scoperto che MoCo danneggia l'apprendimento del modello e che il testo fa particolarmente fatica come modalità. Io tolgo dal modello MoCo visto che chiaramente non mi aiuta.
Pensavo di rifare le ablation sulle modalità per vedere se era colpa di moco

Visto che con i tempi poi probabilmente non ci sto dici di non rifare il tuning degli iperparametri giusto?
In ogni caso la batch size non è più tunabile visto che senza MoCo in locale (macchina che va più veloce) non posso sfruttare l'accumulazione dei batch, restano lr, beta (per kd) e livelli di attention. Teniamo i parametri che abbiamo?