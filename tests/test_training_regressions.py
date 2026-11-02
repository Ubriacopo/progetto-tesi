from types import SimpleNamespace

import pytest
import torch
from torch import nn

from main.dataset.utils import DatasetUidStore
from main.model.blocks.encoder import TemporalEncoder
from main.model.blocks.time_masked import TimeMaskSwitchableProperties
from main.model.downstream.clare.probe.regression.trainer import ClaireTrainer
from main.model.downstream.core.probe_model import Simple1DLinearProbe
from main.model.downstream.core.trainer.base import AbstractCBraClassificationTrainer, AbstractEegAviClassificationTrainer
from main.model.downstream.core.trainer.regression import RegressionTrainer
from main.model.neegavi.train_utils import KdTrainDataModule
from main.model.neegavi.training import EasyEegAviKdVateMaskedModule
from test_model_regressions import tiny_model


def training_module(teacher=None):
    datamodule = SimpleNamespace(dequantize_keys=lambda: [])
    return EasyEegAviKdVateMaskedModule(tiny_model(), teacher, datamodule)


def test_datamodule_constructor_and_data_seed():
    module = KdTrainDataModule([], batch_size=2, seed=17)
    assert module.seed == 17


def test_causal_schedule_uses_estimated_steps_when_max_steps_is_unset():
    module = training_module()
    module._trainer = SimpleNamespace(max_steps=-1, estimated_stepping_batches=100, global_step=50)
    assert module.p_causal_schedule() == pytest.approx(.425)


def test_mode_changes_reach_temporal_support_encoders():
    model = tiny_model()
    temporal = TemporalEncoder(4, 3, 1, TimeMaskSwitchableProperties("bidirectional"), heads=2)
    model.supports[0].adapter = temporal
    model.set_attention_modality(TimeMaskSwitchableProperties("causal"))
    assert temporal.modality.mode == "causal"


def test_teacher_stays_frozen_during_training():
    teacher = nn.Sequential(nn.Linear(4, 4), nn.Dropout(.5))
    teacher.keys = []
    module = training_module(teacher)
    module.train()
    assert not teacher.training
    assert not any(p.requires_grad for p in teacher.parameters())


def test_frozen_probe_stays_in_eval_mode_when_parent_trains():
    backbone = nn.Sequential(nn.Linear(4, 4), nn.Dropout(.5))
    probe = Simple1DLinearProbe(backbone, 4, 1)
    probe.train()
    assert probe.project.training
    assert not backbone.training


def test_test_step_logs_test_losses_instead_of_validation_losses():
    module = training_module()
    observed = []
    module.compute_step_metrics = lambda student, teacher, stage, idx: (observed.append(stage) or {"loss": torch.tensor(0.)})
    batch = {"student/eeg/data": torch.randn(2, 3, 4),
             "student/eeg/mask": torch.ones(2, 3, dtype=torch.bool)}
    module.test_step(batch, 0)
    assert observed == ["test"]
    module.log = lambda *args, **kwargs: None
    module.compute_batch_metrics("test")  # EEG-only: no prefusion support pairs.


class RegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.project = nn.Linear(1, 1)

    def forward(self, batch):
        return batch["assessment", "scores"] * 0 + self.project.weight.flatten()[0]


class TestRegressionTrainer(RegressionTrainer):
    __test__ = False

    def extract_target(self, batch):
        return batch["assessment"]["scores"]


@pytest.mark.parametrize("batch_size,out_dim", [(1, 3), (2, 3), (1, 1), (2, 1)])
def test_regression_accepts_singleton_batches_and_outputs(batch_size, out_dim):
    module = TestRegressionTrainer(RegressionModel(), seed=1, out_dim=out_dim)
    module.log = lambda *args, **kwargs: None
    batch = {"assessment": {"scores": torch.ones(batch_size, out_dim)}}
    assert torch.isfinite(module.training_step(batch))
    assert torch.isfinite(module.validation_step(batch))
    assert torch.isfinite(module.test_step(batch))


def test_clare_tracks_one_output_dimension():
    module = ClaireTrainer(RegressionModel(), seed=1)
    assert module.out_dim == 1
    assert len(module.val_rmse_dims) == 1


def test_finetune_optimizer_has_its_configured_hyperparameters():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.project = nn.Linear(4, 2)
            self.encoder = nn.Linear(4, 4)

    class Trainer(AbstractCBraClassificationTrainer):
        def extract_target(self, batch):
            return batch

    trainer = Trainer(Model(), seed=1, weight_decay=.02)
    assert trainer.configure_optimizers().defaults["weight_decay"] == .02


def test_uid_is_stable_after_reloading_numeric_experiment_ids(tmp_path):
    path = str(tmp_path / "metadata" / "uids.csv")
    first = DatasetUidStore(path).uid("01", "2", "test")
    reloaded = DatasetUidStore(path)
    assert reloaded.uid("01", "2", "test") == first
    assert len(reloaded.df) == 1


def test_faced_trial_labels_use_the_loader_zero_based_indices():
    from tensordict import TensorDict
    from main.model.downstream.faced.datamodule import FacedDataModule

    trials = [0, 3, 12, 16, 27]
    batch = [TensorDict({"meta": {"trial": torch.tensor([trial])},
                         "assessment": {"scores": torch.zeros(1, 3)}}, batch_size=[1])
             for trial in trials]
    result = FacedDataModule.train_collate_fn(batch)
    assert result["assessment", "score"][:, 0].tolist() == [0, 1, 4, 5, 8]


def test_raw_eeg_adapter_converts_time_first_storage_for_cbramod():
    from main.model.neegavi.adapters import EegCbraModAdapter, EegAdapter

    class Encoder(nn.Module):
        def forward(self, x, mask):
            assert x.shape == (1, 2, 3, 200)
            assert mask.shape == (1, 2, 3)
            return x

    adapter = EegCbraModAdapter.__new__(EegCbraModAdapter)
    nn.Module.__init__(adapter)
    adapter.encoder = Encoder()
    adapter.adapter = EegAdapter(2, 200, 4)
    result = adapter(torch.randn(1, 3, 2, 200), torch.ones(1, 3, 2, dtype=torch.bool))
    assert result["data"].shape == (1, 3, 4)


def test_cbramod_baseline_converts_time_first_storage():
    from tensordict import TensorDict
    from main.model.downstream.core.model.base import CBraModBaseModel

    class Encoder(nn.Module):
        def forward(self, x, mask):
            assert x.shape == (2, 2, 3, 200)
            assert mask.shape == (2, 2, 3)
            return x

    class Model(CBraModBaseModel):
        def make_projection_head(self):
            return nn.Identity()

        def freeze(self):
            pass

    model = Model(Encoder(), encoder_frozen=True)
    batch = TensorDict({"eeg": {"data": torch.randn(1, 2, 3, 2, 200),
                                "mask": torch.ones(1, 2, 3, 2, dtype=torch.bool)}}, batch_size=[1, 2])
    assert model(batch).shape == (1, 2, 2, 3, 200)


def test_eegavi_finetune_optimizer_keeps_separate_backbone_learning_rates():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.project = nn.Linear(4, 2)
            self.encoder = nn.ModuleDict({"cbra": nn.Linear(4, 4), "adapter": nn.Linear(4, 4)})

        def get_pivot_adapter(self):
            return SimpleNamespace(encoder=self.encoder["cbra"])

    class Trainer(AbstractEegAviClassificationTrainer):
        def extract_target(self, batch):
            return batch

    trainer = Trainer(Model(), seed=1, classes=2, lr=1e-3, backbone_lr=1e-4, cbramod_lr=1e-5, weight_decay=.02)
    optimizer = trainer.configure_optimizers()
    assert [group["lr"] for group in optimizer.param_groups] == [1e-3, 1e-4, 1e-5]


def test_legacy_faced_trainer_optimizes_unfrozen_backbone_parameters():
    from main.model.downstream.faced.trainer import FacedTrainer

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.project = nn.Linear(4, 2)
            self.backbone = nn.Linear(4, 4)

    model = Model()
    trainer = FacedTrainer(model, labels=2, seed=1)
    optimizer = trainer.configure_optimizers()
    optimized = {id(p) for group in optimizer.param_groups for p in group["params"]}
    assert optimized == {id(p) for p in model.parameters()}


@pytest.mark.parametrize("trainer_name", ["ClaireClassificationTrainer", "CBraModClareClassificationTrainer"])
def test_clare_classification_accepts_frozen_probes_and_scalar_targets(trainer_name):
    from tensordict import TensorDict
    from main.model.downstream.clare import trainer as trainers
    from main.model.downstream.core.probe_model import SimpleCbraLinearProbe

    module = getattr(trainers, trainer_name)(SimpleCbraLinearProbe(200, 2), seed=1)
    assert module.configure_optimizers().param_groups
    target = module.extract_target(TensorDict({"assessment": {"scores": torch.tensor([[4.], [6.]])}}, batch_size=[2]))
    assert target.tolist() == [0, 1]


@pytest.mark.parametrize("pooling", ["linear", "pooling"])
def test_trial_predictions_do_not_depend_on_other_trial_lengths(pooling):
    import tensordict
    from tensordict import TensorDict
    from main.model.downstream.core.probe_model import SimpleLinearProbe, SimplePoolingProbe

    torch.manual_seed(5)
    cls = SimpleLinearProbe if pooling == "linear" else SimplePoolingProbe
    probe = cls(tiny_model("bidirectional"), 4, 1).eval()
    short = TensorDict({"eeg": {"data": torch.randn(1, 3, 4),
                                "mask": torch.ones(1, 3, dtype=torch.bool)}}, batch_size=[1])
    long = TensorDict({"eeg": {"data": torch.randn(2, 3, 4),
                               "mask": torch.ones(2, 3, dtype=torch.bool)}}, batch_size=[2])
    alone = probe(tensordict.pad_sequence([short], 0))[0]
    together = probe(tensordict.pad_sequence([short, long], 0))[0]
    torch.testing.assert_close(alone, together)
