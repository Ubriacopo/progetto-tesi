- Dataset implementation + Dataloader
- Run download script again
- Prepare model for distillation
    - EEG on intermediate layer
    - VATE on intermediate layer
    - final label comp contrastive learning (Clustering into classes) (TIPO ABAE)
    -

> Hybrid policy: per video, use K_uniform + K_eeg clips (e.g., 6 + 6 for a 2-min video).
> Train with both so the model doesn’t depend on EEG being present.

> Potrei scomporre la classe Compose con due metodi di call (uno transfomr e uno augment)
> Questa opzione sembra "papabile" -> potremmo provarci dai.
> transform() - augment() 
> call() -> tuple[torch.Tensor, torch.Tensor] (transform,  tranform + augment) as res
