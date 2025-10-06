from __future__ import annotations

from typing import Callable

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from model.loss import siglip
from model.EEGAVI.EEGAVI import EEGAVI
from model.VATE.constrastive_model import ContrastiveModel


class EEGAVIVateTrainer:
    @staticmethod
    def default_kd_vate_trainer(model: tuple[str, EEGAVI]) -> EEGAVIVateTrainer:
        return EEGAVIVateTrainer(
            model=model,
            teacher=("VATE", ContrastiveModel(200, 100)),

            # TODO Tailor it for the model
            loss_function=siglip,

            optimizer_constructor=Adam,
            optimizer_args={'lr': 1e-3},
        )

    def __init__(self, model: tuple[str, EEGAVI], teacher: tuple[str, ContrastiveModel],
                 loss_function: Callable, optimizer_constructor: Callable, optimizer_args: dict, ):
        self.student_key, self.model = model
        self.teacher_key, self.teacher = teacher
        # Initialize the optimizer for the current trainer
        self.optimizer = optimizer_constructor(self.model.parameters(), **optimizer_args)
        self.loss_fn = loss_function

    def train(self, data_loader: DataLoader, epochs: int):
        self.model.train()

        for epoch in range(epochs):
            running_loss = .0
            # x_s -> x for student | x_t -> x for teacher
            for x in data_loader:
                x_t = x[self.teacher_key]
                x_s = x[self.student_key]

                self.optimizer.zero_grad()
                # Teacher is frozen.
                with torch.no_grad():
                    # TODO Vedere storia di masking e cosa passare. Forse mi basta in ds transform per VATE fare una map a None
                    y_t = self.teacher(x_t["vid"], x_t["aud"], x_t["text"])

                y_s, kd_s = self.model(x_s)

                loss = self.loss_fn(y_s, kd_s, y_t)
                loss.backward()
                running_loss += loss.item()

                # Release memory? TODO Vedi se si fa di solito
                del x_t
                del x_s
                del kd_s
                del y_s
                del y_t
                torch.cuda.empty_cache()

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len()}")

    def test(self):
        raise NotImplementedError()
