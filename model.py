import pytorch_lightning as pl
import timm
import torch
from torch import nn
from sklearn.metrics import confusion_matrix
import wandb
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class ImageClassifier(pl.LightningModule):
    def __init__(self, NUM_CLASSES, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = timm.create_model(model_name='resnet34', pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, NUM_CLASSES)
        self.criterion = nn.CrossEntropyLoss()
        self.train_losses = []
        self.train_accs = []
        self.val_losses = []
        self.val_accs = []
        self.val_preds = []
        self.val_labels = []

    def forward(self, xb):
        return self.model(xb)

    def training_step(self, batch, batch_idx):
        xb, yb = batch
        logits = self(xb)
        loss = self.criterion(logits, yb)
        acc = (logits.argmax(dim=1) == yb).float().mean()
        self.train_losses.append(loss)
        self.train_accs.append(acc)
        self.log('train_loss', loss, prog_bar=True, on_step=True)
        self.log('train_acc', acc, prog_bar=True, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        xb, yb = batch
        logits = self(xb)
        loss = self.criterion(logits, yb)
        acc = (logits.argmax(dim=1) == yb).float().mean()
        preds = logits.argmax(dim=1)
        self.val_preds.append(preds)
        self.val_labels.append(yb)
        self.val_losses.append(loss)
        self.val_accs.append(acc)
        self.log('val_loss', loss, prog_bar=True, on_step=True)
        self.log('val_acc', acc, prog_bar=True, on_step=True)
        return loss

    def on_train_epoch_end(self):
        avg_train_loss = torch.stack(self.train_losses).mean()
        avg_train_acc = torch.stack(self.train_accs).mean()
        self.log('avg_train_loss', avg_train_loss,
                 prog_bar=True, on_epoch=True)
        self.log('avg_train_acc', avg_train_acc, prog_bar=True, on_epoch=True)
        self.train_losses.clear()
        self.train_accs.clear()

    def on_validation_epoch_end(self):
        avg_val_loss = torch.stack(self.val_losses).mean()
        avg_val_acc = torch.stack(self.val_accs).mean()
        self.log('avg_val_loss', avg_val_loss, prog_bar=True, on_epoch=True)
        self.log('avg_val_acc', avg_val_acc, prog_bar=True, on_epoch=True)

        # Compute confusion matrix
        preds = torch.cat(self.val_preds).cpu().numpy()
        labels = torch.cat(self.val_labels).cpu().numpy()

        # Log the WandB confusion matrix plot
        if self.logger:
            self.logger.experiment.log({
                "confusion_matrix": wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=labels,
                    preds=preds,
                    class_names=[f"Class {i}" for i in range(
                        self.hparams.NUM_CLASSES)]
                ),
                "epoch": self.current_epoch
            })

        # Clear the lists for the next epoch
        self.val_losses.clear()
        self.val_accs.clear()
        self.val_preds.clear()
        self.val_labels.clear()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
