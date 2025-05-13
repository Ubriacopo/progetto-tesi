## Next steps:

- [ ] Work on the datasets and pre-process if needed.
    - [ ] We want to use the maximum amount of data possible. We also need to have coherent
     representations between EEG in input. Video processing?
    - [ ] Design a pre processing pipeline
- [ ] Try the BrainBERT on the other datasets.
- [ ] Try to run also NeuroGPT
- [ ] Fork the BrainBERT repository and update it in order to work as I did currently.
- [ ] Run embeddings of VATE on our current data

# Idea:
Devo apprendere embeddings unificati di DEAP e AMIGOS.
> So masking on missing features is the go to?

1. Define a unified channel set (e.g., DEAP's 32 channels):
For DEAP: use all 32 channels
For AMIGOS: map available channels to the 32, pad missing ones with zeros

2. Add a binary mask (1 for present channels, 0 for padded)
Pass it into the model alongside the data (as attention or feature mask)

3. Resample signals to same frequency (e.g., 128 Hz)
Use fixed time windows (e.g., 3 seconds → 384 time points)




