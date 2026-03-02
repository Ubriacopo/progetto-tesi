# todo this next
import optuna
import torch
from main.app_config import AppConfig
from main.core_data.dataset import CachableDatasetDescriptor
from main.model.VATE.constrastive_model import MaskedContrastiveModel
from main.model.neegavi.config import EegModalityConfig, MaskedFeedForwardConfig, KdPerceiverModalityConfig
from main.model.neegavi.factory import Factory
from main.model.neegavi.model import EegInterAviModelConfiguration
from main.model.neegavi.train_utils import KdTrainDataModule
from main.model.neegavi.training import EasyEegAviKdVateMaskedModule

# todo cerca di tenere le cose il piu posisibli semplici
def objective(
        trial: optuna.Trial,

        teacher: MaskedContrastiveModel,
        dataset_descriptors: list[CachableDatasetDescriptor],

        eeg_config: EegModalityConfig,
        vid_config: KdPerceiverModalityConfig,
        aud_config: KdPerceiverModalityConfig,
        txt_config: KdPerceiverModalityConfig,
        ecg_config: MaskedFeedForwardConfig,
        custom_config: EegInterAviModelConfiguration,
        drop_p_min: float = 0.05,
        drop_p_max: float = 0.2,
        attention_max_layers: int = 4,
        attention_min_layers: int = 2,
):
    torch.manual_seed(AppConfig.SEED)  # Reproducibility
    # Tuned grid of parameters
    attn_layers = trial.suggest_int(name="attn_layers", low=attention_min_layers, high=attention_max_layers, step=1)
    drop_p = trial.suggest_float(name="drop_p", low=drop_p_min, high=drop_p_max, step=0.05)
    batch_size = trial.suggest_categorical(name="batch_size", choices=[32, 64, 128])
    use_moco = trial.suggest_categorical(name="use_moco", choices=[True, False])

    custom_config.drop_p = drop_p
    student = Factory.default(
        eeg_config=eeg_config,
        vid_config=vid_config,
        aud_config=aud_config,
        txt_config=txt_config,
        ecg_config=ecg_config,
        attention_config=attn_layers,  # Simple is strong, just choose how many to stack togheter
        custom_config=custom_config,
    )

    datamodule = KdTrainDataModule(
        dataset_paths=dataset_descriptors,
        batch_size=batch_size,
        dequantize_keys=["eeg", "aud", "vid", "txt", "ecg"],
        seed=AppConfig.SEED
    )

    module = EasyEegAviKdVateMaskedModule(
        student=student,
        teacher=teacher,
        datamodule=datamodule,
        use_moco=use_moco,
    )
