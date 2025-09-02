from torch.utils.data import StackDataset

from common.data.data_point import EEGDatasetTransformWrapper
from common.data.dataset import KDEEGPdSpecMediaDataset


def kd_train_dataset(amigos_path: str):
    return StackDataset(
        KDEEGPdSpecMediaDataset(
            amigos_path,
            shared_transform=EEGDatasetTransformWrapper(
                name="shared_transform",
                vid_transform=[

                ],
                aud_transform=[

                ],
                eeg_transform=[

                ]
            ),
            modality_transforms=[
                EEGDatasetTransformWrapper(
                    name="EEGAVI",
                    vid_transform=[

                    ],
                    aud_transform=[

                    ],
                    eeg_transform=[

                    ],
                )
            ]
        )
    )
