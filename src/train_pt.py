import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
from dataset import CData
from torch.utils.data import DataLoader
from model import ImageClassifier
import wandb
import time
import os
from datetime import datetime
from engine import train_model
import torch
from dataclasses import dataclass
from dotenv import load_dotenv
import random
import numpy as np


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class CFG:
    path: Path = Path('data')
    TARGET_SIZE: int = 224
    BATCH_SIZE: int = 32
    NUM_EPOCHS: int = 2
    use_wandb: bool = True
    PROJECT_NAME: str = "cassava-classifier"
    SAMPLE_SIZE: int = 200
    SEED: int = 42
    device: str = 'cuda' if torch.device.is_available() else 'cpu'


# Load and split data
load_dotenv()
cfg = CFG()
seed_everything(cfg.SEED)
df = pd.read_csv(cfg.path / 'train.csv')
if cfg.SAMPLE_SIZE is not None:
    df = df.sample(cfg.SAMPLE_SIZE)
train_df, valid_df = train_test_split(df, test_size=0.2, stratify=df['label'])

# Calculate dataset sizes
train_size = len(train_df)
valid_size = len(valid_df)

# Create run name with dataset sizes
run_name = f"cassava_train{train_size}_val{valid_size}_{
    datetime.now().strftime('%Y%m%d')}"
if cfg.use_wandb:
    logger = wandb.init(
        project=cfg.PROJECT_NAME,
        name=run_name,
        config=vars(cfg)
    )


# Define transforms
train_transform = A.Compose([
    A.SmallestMaxSize(max_size=cfg.TARGET_SIZE, p=1.0),
    A.RandomCrop(height=cfg.TARGET_SIZE, width=cfg.TARGET_SIZE, p=1.0),
    A.HorizontalFlip(p=0.5),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])
val_transform = A.Compose([
    A.SmallestMaxSize(max_size=cfg.TARGET_SIZE, p=1.0),
    A.CenterCrop(height=cfg.TARGET_SIZE, width=cfg.TARGET_SIZE, p=1.0),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

# Create datasets and dataloaders
train_ds = CData(train_df, cfg.path / 'train_images', train_transform)
valid_ds = CData(valid_df, cfg.path / 'train_images', val_transform)
train_dl = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE,
                      shuffle=True, num_workers=os.cpu_count(), pin_memory=True)
valid_dl = DataLoader(valid_ds, batch_size=cfg.BATCH_SIZE,
                      shuffle=False, num_workers=os.cpu_count(), pin_memory=True)

# Initialize model
model = ImageClassifier(model_name='resnet18', num_classes=5)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

tik = time.time()
train_model(model, train_dl, valid_dl, optimizer, cfg.device, cfg.NUM_EPOCHS)
tok = time.time()
print(f'time taken {tok-tik:.2}s')
if cfg.use_wandb:
    wandb.finish()
