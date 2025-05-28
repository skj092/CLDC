import pytorch_lightning as pl
import timm
import torch
from torch import nn


class ImageClassifier(pl.LightningModule):
    def __init__(self, NUM_CLASSES, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = timm.create_model(model_name='resnet34')
        self.model.fc = nn.Linear(self.model.fc.in_features, NUM_CLASSES)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, xb):
        return self.model(xb)

    def training_step(self, batch, batch_idx):
        xb, yb = batch
        logits = self(xb)
        loss = self.criterion(logits, yb)
        acc = (logits.argmax(dim=1) == yb).float().mean()
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        xb, yb = batch
        logits = self(xb)
        loss = self.criterion(logits, yb)
        acc = (logits.argmax(dim=1) == yb).float().mean()
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
