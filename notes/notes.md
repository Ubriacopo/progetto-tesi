# Sampling notes

Long recordings need both coverage and informative segments.

Broader limitations and proposed solutions are collected in [`future-work.md`](future-work.md).

## Preferred strategy

Candidate windows are sampled uniformly, then selection is biased toward EEG changes such as variance peaks or threshold crossings. Uniform sampling provides the comparison baseline.

Short sequences are padded with zeros and paired with an explicit validity mask.

Video may be downsampled before encoding. The effective duration and timestamps remain aligned with the EEG windows.

## Reproducible shuffling

Each sampler receives an explicit generator:

```python
generator = torch.Generator().manual_seed(1)
sampler = RandomSampler(dataset, generator=generator)
loader = DataLoader(dataset, sampler=sampler)
```

Postprocessing checks establish whether each expected sample exists in every required modality.
