import torch


class EegCanonicalOrder:
    order: list[str] = [
        # First 14 channels are dominated by AMIGOS as it has less
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
        "T7",
        "CP5",
        "CP1",
        "P3",
        "PO3"
        "Oz",
        "Pz",
        "Fp2",
        "AF4",
        "Fz",
        "F8",
        "FC6",
        "FC2",
        "Cz",
        "C4",
        "T8",
        "CP6",
        "CP2",
        "P4",
        "P8",
        "PO4",
        "O2"
    ]

    def adapt(self, eeg: torch.Tensor, tensor_order: list[str]):
        sorted_object = []
        for idx, entry in enumerate(tensor_order):
            i = self.order.index(entry)
            sorted_object[i] = eeg[idx]

        return_tensor = torch.zeros(len(self.order), *eeg.shape[1:], device=eeg.device)
        mask = torch.zeros(len(self.order), device=eeg.device).bool()

        sorted_object = torch.tensor(sorted_object, device=eeg.device, dtype=eeg.dtype)
        # Pad to maximum length if we are below it.
        return_tensor[:sorted_object.shape[0]] = sorted_object
        mask[:sorted_object.shape[0]] = True

        return return_tensor, mask
