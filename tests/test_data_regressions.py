import random
from types import SimpleNamespace

import h5py
import mne
import numpy as np
import pandas as pd
import torch

from main.core_data.dataset import H5KdDataset
from main.core_data.shards import FusedDataSharder, KdDataSharder
from main.core_data.media.assessment.assessment import Assessment
from main.core_data.media.assessment.transform import RescaleAssessmentValue
from main.core_data.media.signal.transforms import SubclipMneRaw


def test_float_label_rescaling_changes_values_without_changing_other_trials():
    original = Assessment(eid=1, data=np.array([2, 3, 4]), labels=["a", "v", "d"], scales=[(1, 5)] * 3)
    cloned = original.clone({"eid": 2})
    RescaleAssessmentValue("a", (1., 9.))(cloned)
    assert cloned.data[0] == 3.
    assert original.data[0] == 2
    assert original.scales[0] == (1, 5)
    fractional = Assessment(eid=1, data=np.array([2]), labels=["a"], scales=[(1, 5)])
    RescaleAssessmentValue("a", (0., 1.))(fractional)
    assert fractional.data[0] == .25


def test_cropping_a_segment_does_not_truncate_the_source_recording():
    raw = mne.io.RawArray(np.arange(1000)[None], mne.create_info(["eeg"], 100, ["eeg"]), verbose=False)
    first = SimpleNamespace(data=raw, interval=(0., 2.))
    second = SimpleNamespace(data=raw, interval=(5., 7.))
    SubclipMneRaw()(first)
    SubclipMneRaw()(second)
    assert raw.n_times == 1000
    assert second.data.get_data()[0, 0] == 500


def write_shard(path, n=5):
    with h5py.File(path, "w") as h5:
        h5.attrs["num_samples"] = n
        h5.create_dataset("student/eeg/data", data=np.arange(n)[:, None])


def test_hdf5_iteration_counts_partial_batches_and_preserves_samples(tmp_path):
    write_shard(tmp_path / "train_000.h5")
    ds = H5KdDataset(str(tmp_path), "train", batch_size=3, buffer_size=4, block_size=2, shuffle=False, take_keys=[])
    batches = list(ds)
    assert len(ds) == len(batches) == 2
    values = torch.cat([batch["student/eeg/data"] for batch in batches])
    assert values[:, 0].tolist() == list(range(5))


def test_block_shuffling_uses_its_own_rng():
    block = {"data": torch.arange(20)}
    buffers = [[], []]
    for i in range(2):
        torch.manual_seed(i * 100)
        H5KdDataset.add_block(None, 0, block, buffers[i], random.Random(42))
    assert torch.equal(buffers[0][0][1], buffers[1][0][1])


def test_fractional_loading_does_not_load_an_entire_large_shard(tmp_path):
    write_shard(tmp_path / "train_000.h5", n=20)
    ds = H5KdDataset(str(tmp_path), "train", batch_size=2, buffer_size=4,
                     block_size=2, shuffle=False, limit_data=.25)
    assert sum(batch["student/eeg/data"].shape[0] for batch in ds) == 5


def test_shard_rollover_resets_dataset_capacity(tmp_path):
    spec = tmp_path / "spec.csv"
    pd.DataFrame({"eid": [0], "person_id": [0], "index": [0]}).to_csv(spec, index=False)
    sharder = FusedDataSharder(str(spec), str(tmp_path / "out"), chunk0=1)
    first, _, next_id, _ = sharder.open_new_shard(None, 0, 0, "train")
    sharder.do_append(first, {"data": torch.ones(2)}, "student/eeg", 0)
    second, _, _, _ = sharder.open_new_shard(first, next_id, 1, "train")
    try:
        sharder.do_append(second, {"data": torch.ones(2) * 2}, "student/eeg", 0)
        assert second["student/eeg/data"][0].tolist() == [2., 2.]
    finally:
        sharder.close_dandling_shard(second, 1)


def test_legacy_sharder_keeps_participants_in_one_split():
    sharder = KdDataSharder.__new__(KdDataSharder)
    sharder.val_participants = sharder.test_participants = 1
    sharder.student_df = pd.DataFrame([
        {"person_id": person, "eid": person * 10 + trial, "index": window}
        for person in range(4) for trial in range(2) for window in range(3)
    ])
    splits = sharder.extract_rows_by_eid()
    subjects = [{row.person_id for records in split.values() for row in records} for split in splits]
    assert len(subjects[0]) == 2
    assert all(subjects[i].isdisjoint(subjects[j]) for i in range(3) for j in range(i))


def test_branch_preserves_names_and_calls_each_module():
    from torch import nn
    from main.core_data.processing.transform import Branch

    result = Branch(("same", nn.Identity()), ("negative", nn.Identity()))(torch.ones(2))
    assert set(result) == {"same", "negative"}
    assert result["same"].eq(1).all()


def test_empty_unmasked_fallback_returns_the_requested_shape():
    from main.core_data.processing.transform import EmptyObjectTransform

    result = EmptyObjectTransform((2, 3))()
    assert result.shape == (2, 3)
    assert result.eq(0).all()
