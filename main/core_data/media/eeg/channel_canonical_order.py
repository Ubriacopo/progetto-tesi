import torch


class EegCanonicalOrder:
    order: list[str] = [
        # 10 - 20
        "Fp1",
        "Fp2",
        "F7",
        "F3",
        "Fz",
        "F4",
        "F8",
        "T7",  # T3
        "C3",
        "Cz",
        "C4",
        "T8"  # T4, 
        "P7"  # T5,
        "P3",
        "Pz",
        "P4",
        "P8"  # T6,
        "O1",
        "O2"
    ]

    def __init__(self):
        self.order_to_index = {ch.lower(): i for i, ch in enumerate(self.order)}

    def adapt(self, eeg: torch.Tensor, tensor_order: list[str]):
        return_tensor = torch.zeros(len(self.order), *eeg.shape[1:], device=eeg.device)
        mask = torch.zeros(*return_tensor.shape[:-1], device=eeg.device)

        for current_idx, entry in enumerate(tensor_order):
            if not entry.lower() in self.order_to_index:
                continue  # Non valid elements are not tracked

            new_idx = self.order_to_index[entry.lower()]
            return_tensor[new_idx] = eeg[current_idx]
            mask[new_idx] = 1

        return return_tensor, mask
