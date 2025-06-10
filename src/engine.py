from tqdm import tqdm
import numpy as np
import wandb
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def train_model(model, train_loader, val_loader, optimizer, device, num_epochs=10):
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        train_losses, train_accs = [], []

        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch}"):
            batch = [b.to(device) for b in batch]
            loss, acc = model.training_step(batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            train_accs.append(acc.item())

        avg_train_loss = np.mean(train_losses)
        avg_train_acc = np.mean(train_accs)

        model.eval()
        val_losses, val_accs = [], []
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Validation Epoch {epoch}"):
                batch = [b.to(device) for b in batch]
                loss, acc, preds, labels = model.validation_step(batch)

                val_losses.append(loss.item())
                val_accs.append(acc.item())
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())

        avg_val_loss = np.mean(val_losses)
        avg_val_acc = np.mean(val_accs)

        # Print metrics to console
        print(f"\nEpoch {epoch}")
        print(f"Train Loss: {avg_train_loss:.4f} | Train Acc: {
              avg_train_acc:.4f}")
        print(f"Val   Loss: {avg_val_loss:.4f} | Val   Acc: {avg_val_acc:.4f}")

        # Log to wandb
        wandb.log({
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'train_acc': avg_train_acc,
            'val_loss': avg_val_loss,
            'val_acc': avg_val_acc
        })

        # Confusion matrix
        preds = torch.cat(all_preds).numpy()
        labels = torch.cat(all_labels).numpy()
        cm = confusion_matrix(labels, preds, labels=range(5))
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(f'Confusion Matrix - Epoch {epoch}')
        wandb.log({f'confusion_matrix_epoch_{epoch}': wandb.Image(fig)})
        plt.close(fig)
