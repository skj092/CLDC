import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
from dataset import CData
from torch.utils.data import DataLoader
from model import ImageClassifier
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
import wandb
import time
import os
from datetime import datetime


path = Path('data')
TARGET_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 5
use_wandb = True
SAMPLE_SIZE = 100  # Configurable sample size for experimentation (set to None for full dataset)
seed_everything(42)

# Load and split data
df = pd.read_csv(path / 'train.csv')
if SAMPLE_SIZE is not None:
    df = df.sample(SAMPLE_SIZE)
train_df, valid_df = train_test_split(df, test_size=0.2, stratify=df['label'])

# Calculate dataset sizes
train_size = len(train_df)
valid_size = len(valid_df)

# Create run name with dataset sizes
run_name = f"cassava_train{train_size}_val{valid_size}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# Define transforms
train_transform = A.Compose([
    A.SmallestMaxSize(max_size=TARGET_SIZE, p=1.0),
    A.RandomCrop(height=TARGET_SIZE, width=TARGET_SIZE, p=1.0),
    A.HorizontalFlip(p=0.5),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])
val_transform = A.Compose([
    A.SmallestMaxSize(max_size=TARGET_SIZE, p=1.0),
    A.CenterCrop(height=TARGET_SIZE, width=TARGET_SIZE, p=1.0),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

# Create datasets and dataloaders
train_ds = CData(train_df, path / 'train_images', train_transform)
valid_ds = CData(valid_df, path / 'train_images', val_transform)
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=os.cpu_count(), pin_memory=True)
valid_dl = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=os.cpu_count(), pin_memory=True)

# Initialize model
model = ImageClassifier(5, lr=1e-3)

# Set up logger and callbacks
logger = WandbLogger(project="cassava-leaf-disease", log_model=False, name=run_name) if use_wandb else None

# Log dataset sizes to WandB
if logger:
    wandb.config.update({
        "train_dataset_size": train_size,
        "valid_dataset_size": valid_size,
        "sample_size": SAMPLE_SIZE if SAMPLE_SIZE is not None else "full",
    })

callbacks = [ModelCheckpoint(monitor='val_acc', mode='max', save_top_k=1)]

# Configure Trainer for multi-GPU with DDP
trainer = Trainer(
    max_epochs=NUM_EPOCHS,
    logger=logger,
    callbacks=callbacks,
    accelerator='auto',
    devices='auto',
)

# Train the model
tik = time.time()
trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=valid_dl)
tok = time.time()
print(f"Total time taken {tok-tik:.2f}s")

if logger:
    wandb.finish()
