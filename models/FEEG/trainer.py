import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from models.FEEG.model import EEGAVI
from models.VATE.constrastive_model import ContrastiveModel


# todo
# ad hoc for this instance?
class EEGAVIKDTrainer:
    def __init__(self, model: EEGAVI, teacher: ContrastiveModel, loss_fn, learning_rate: float = 1e-3):
        self.model: EEGAVI = model
        self.teacher: ContrastiveModel = teacher

        self.loss_fn = loss_fn
        self.optimizer: torch.optim.Optimizer = Adam(self.model.parameters(), lr=learning_rate)

    # todo train loader deve contenre anche records per teacher
    def train(self, train_loader: DataLoader, epochs: int = 20):
        self.model.train()
        for epoch in range(epochs):
            running_loss = .0
            for x, teacher_x in train_loader:
                self.optimizer.zero_grad()

                with torch.no_grad():
                    logits = self.teacher(teacher_x)

                outputs, kd_objects = self.model(x)
                # TODO: I can apply CLIP-loss
                loss = self.loss_fn(outputs, x)

                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader)}")
