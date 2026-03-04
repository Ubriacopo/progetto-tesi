import torch


class EegCanonicalOrder:
    order: list[str] = [
        # First 14 channels are dominated by AMIGOS and DREAMER as it they have less (Emotiv EPOC)
        # Because the 10–20 system was renamed over time. The electrodes are the same physical positions, but the labels changed between older and newer conventions
        # This is for those noted with eq(other) (legacy labels)
        "AF3",  # Not in CBra training
        "F7",
        "F3",
        "FC5",  # Not in CBra training
        "T7",  # eq(T3) not in CBra
        "P7",  # eq(T5)
        "O1",
        "O2",
        "P8",  # eq(T6)
        "T8",  # eq(T4)
        "FC6",  # Not in CBra training
        "F4",
        "F8",
        "AF4"  # Not in CBra training,
        # DEAP rest
        "FP1",
        "FC1",  # Not in CBra training
        "C3",
        "CP5",  # Not in CBra training
        "CP1",  # Not in CBra training
        "P3",
        "PO3",
        "Oz",  # Not in CBra training
        "Pz",
        "Fp2",
        "Fz",
        "FC2",  # Not in CBra training
        "Cz",
        "C4",
        "CP6",  # Not in CBra training
        "CP2",  # Not in CBra training
        "P4",
        "PO4",  # Not in CBra training
        # Coming From EAV
        "PO9",  # Not in CBra training
        "PO10"  # Not in CBra training
    ]

    def __init__(self):
        self.order_to_index = {ch.lower(): i for i, ch in enumerate(self.order)}

    def adapt(self, eeg: torch.Tensor, tensor_order: list[str]):
        return_tensor = torch.zeros(len(self.order), *eeg.shape[1:], device=eeg.device)
        mask = torch.zeros(*return_tensor.shape[:-1], device=eeg.device)

        for current_idx, entry in enumerate(tensor_order):
            new_idx = self.order_to_index[entry.lower()]
            return_tensor[new_idx] = eeg[current_idx]
            mask[new_idx] = 1

        return return_tensor, mask
