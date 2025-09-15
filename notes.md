> For long EEG-video sequences (3–4 minutes), here's how to approach sampling:

Hybrid:
Apply random sampling with bias toward segments with EEG activity changes (e.g., threshold crossings, variance spikes).

Balance coverage and informativeness.
A questo potresti applicare sliding window.

Se troppo difficile: Uniform Random Sampling (Baseline)

Zero padding su elementi che non arrivano a lunghezza desiderata

Downsampling can be applied to video (if video is in 30fps I compress the fps ex 15fps video -> 2 seconds from 1 second)
> This approach is commonly used in VideoMAE, TimeSformer, and X-CLIP as well.<br>
> So yes—downsampling and adjusting frame rate is a standard and smart move.


Per risolvere problema di random con shuffle: (Così posso tenere i due dataset disinti a patto abbia stesso num di
samples che è ovviamente così) -> TODO: Pulizia postprocessing per vedere che esistono tutti i sample.

```py
G = torch.Generator()
G = G.manual_seed(1)

ran_sampler = RandomSampler(data_source=torch.ones(20), generator=G)
loader = DataLoader(ds, sampler=ran_sampler)
```