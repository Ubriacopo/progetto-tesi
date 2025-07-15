import torch


class SimpleLoaderDataset(torch.utils.data.Dataset):
    def __init__(self, folds: list[list]):
        self.folds = folds

    def __getitem__(self, item: int):
        # Se manca un testo posso ricevere [num, num, None, num]. Multimodal safe?
        return tuple(lst[item] if len(lst) > item else None for lst in self.folds)

    def __len__(self):
        return len(self.folds[0])
