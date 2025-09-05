from abc import ABC, abstractmethod
from typing import Callable

import torch
from torch import nn
from torch.utils.data import DataLoader

from models.EEGAVI.EEGAVI import EEGAVI


class EEGAVITeacherSingleTeacher:

    def __init__(self,
                 model: tuple[str, EEGAVI], teacher: tuple[str, nn.Module],
                 loss_function: Callable, optimizer_constructor: Callable, optimizer_args: dict, ):
        self.student_key, self.model = model
        self.teacher_key, self.teacher = teacher
        # Initialize the optimizer for the current trainer
        self.optimizer = optimizer_constructor(self.model.parameters(), **optimizer_args)
        self.loss_fn = loss_function

    # TODO Non farlo troppo generale perchè ciao altrimenti che complichiamo inutilmente le cose
    #           -> No classe astratta e farei optimizer e loss passate in costruzione.
    #               COMPOSITION OVER INHERITANCE
    def train(self, dataloader: DataLoader, epochs: int):
        self.model.train()

        for epoch in range(epochs):
            running_loss = .0
            # x_s -> x for student | x_t -> x for teacher
            for x in dataloader:
                x_t = x[self.teacher_key]
                x_s = x[self.student_key]

                self.optimizer.zero_grad()

                # Teacher is frozen.
                with torch.no_grad():
                    y_t = self.teacher(x_t)
                y_s, kd_s = self.model(x_s)

                loss = self.loss_fn(y_s, kd_s, y_t)
                loss.backward()
                running_loss += loss.item()

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(dataloader)}")

    def test(self):
        pass
