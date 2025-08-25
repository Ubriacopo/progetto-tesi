import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from models.FEEG.model import EEGAVI


# ad hoc for this instance?
class EEGAVIKDTrainer:
    def __init__(self, model: EEGAVI, teacher: nn.Module, loss_fn, learning_rate: float = 1e-3):
        self.model: EEGAVI = model
        self.teacher: nn.Module = teacher
        self.loss_fn = loss_fn
        self.optimizer: torch.optim.Optimizer = Adam(self.model.parameters(), lr=learning_rate)

    def train(self, train_loader: DataLoader, val_loader: DataLoader = None, epochs: int = 20):
        self.model.train()
        for epoch in range(epochs):
            running_loss = .0
            for x in train_loader:
                self.optimizer.zero_grad()

                with torch.no_grad():
                    x_teacher = self.adapt_input_for_teacher()
                    logits = self.teacher(x_teacher)

                outputs, kd_objects = self.model(x)
                # TODO
                loss = self.loss_fn(outputs, x)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader)}")

    def adapt_input_for_teacher(self):
        pass

    def test(self):
        pass
