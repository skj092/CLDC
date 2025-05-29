import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import albumentations as A
from dataset import CData
from torch.utils.data import DataLoader
from model import ImageClassifier
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
import wandb
import os
import time

path = Path('data')
TARGET_SIZE = 224  # Example input size
BATCH_SIZE = 32
NUM_EPOCHS = 1
use_wandb = False
seed_everything(42)

df = pd.read_csv(path/'train.csv').sample(500)
train_df, valid_df = train_test_split(df, test_size=0.2, stratify=df['label'])
train_transform = A.Compose([
    A.SmallestMaxSize(max_size=TARGET_SIZE, p=1.0),
    A.RandomCrop(height=TARGET_SIZE, width=TARGET_SIZE, p=1.0),
    A.HorizontalFlip(p=0.5),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    A.ToTensorV2(),
])
val_transform = A.Compose([
    A.SmallestMaxSize(max_size=TARGET_SIZE, p=1.0),
    A.CenterCrop(height=TARGET_SIZE, width=TARGET_SIZE, p=1.0),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    A.ToTensorV2(),
])

train_ds = CData(train_df, path/'train_images', train_transform)
valid_ds = CData(valid_df, path/'train_images', train_transform)
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE,
                      shuffle=True, num_workers=os.cpu_count())
valid_dl = DataLoader(valid_ds, batch_size=BATCH_SIZE,
                      shuffle=False, num_workers=os.cpu_count())

model = ImageClassifier(5, lr=1e-3)
logger = WandbLogger(project="cassava-leaf-disease",
                     log_model=False) if use_wandb else None
callbacks = [ModelCheckpoint(monitor='val_acc', mode='max', save_top_k=1)]
trainer = Trainer(max_epochs=NUM_EPOCHS, logger=logger,
                  callbacks=callbacks, accelerator='auto', devices='auto')

tik = time.time()
trainer.fit(model, train_dataloaders=train_dl,
            val_dataloaders=valid_dl)
tok = time.time()
print(f"Total time taken {tok-tik:.2}s")
if logger:
    wandb.finish()
