# ECG representation

ECG features are extracted with [ECG-FM](https://github.com/bowang-lab/ECG-FM). It was selected because it is established and has been evaluated under missing-lead conditions.

Alternatives considered during development:

- [TolerantECG](https://github.com/Fsoft-AIC/TolerantECG), which targets incomplete lead sets.
- [ECGFounder](https://github.com/PKUDigitalHealth/ECGFounder), which provides a useful baseline.

AMIGOS exposes RA and LA signals. Their difference approximates lead I. RA and LL can also form lead II when both are available. Dataset adapters must document any derived lead mapping.
