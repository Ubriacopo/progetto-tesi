import torch


class EegCanonicalOrder:
    order: list[str] = [
        # First 14 channels are dominated by AMIGOS and DREAMER as it they have less (Emotiv EPOC)
        "AF3",
        "F7",
        "F3",
        "FC5",
        "T7",
        "P7",
        "O1",
        "O2",
        "P8",
        "T8",
        "FC6",
        "F4",
        "F8",
        "AF4",
        # DEAP rest
        "Fp1",
        "FC1",
        "C3",
        "CP5",
        "CP1",
        "P3",
        "PO3",
        "Oz",
        "Pz",
        "Fp2",
        "Fz",
        "FC2",
        "Cz",
        "C4",
        "CP6",
        "CP2",
        "P4",
        "PO4",
    ]

    def adapt(self, eeg: torch.Tensor, tensor_order: list[str]):
        return_tensor = torch.zeros(len(self.order), *eeg.shape[1:], device=eeg.device)
        mask = torch.zeros(*return_tensor.shape[:-1], device=eeg.device)

        for current_idx, entry in enumerate(tensor_order):
            new_idx = self.order.index(entry)
            return_tensor[new_idx] = eeg[current_idx]
            mask[new_idx] = 1

        return return_tensor, mask
