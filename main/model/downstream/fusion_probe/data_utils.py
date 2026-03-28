import tensordict
from tensordict import TensorDict


def train_collate(batch: list[TensorDict]):
    batch = [b.exclude("meta", ("assessment", "scales"), ("assessment", "labels"), )[:10] for b in batch]
    for td in batch:
        # Only take the first 5 (V/A/D/L/F) because the train dataset (AMIGOS) has more
        td["assessment", "scores"] = td["assessment", "scores"][:, :5]

    return tensordict.pad_sequence(batch, 0, return_mask="pad_mask")


def test_collate_fn(batch: list[TensorDict]):
    batch = [b.exclude("meta", ("assessment", "scales"), ("assessment", "labels"), )[:10] for b in batch]
    return tensordict.pad_sequence(batch, 0, return_mask="pad_mask")
