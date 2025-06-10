from torch.utils.data import Dataset
import os
from pathlib import Path
import pandas as pd
import torch
import cv2


class CData(Dataset):
    def __init__(self, df, root_dir, transforms=None):
        self.df = df
        self.root_dir = root_dir
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_id, label = self.df.iloc[idx]
        img = cv2.imread(os.path.join(self.root_dir, img_id))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transforms:
            img = self.transforms(image=img)['image']
        return img, torch.tensor(label)


if __name__ == "__main__":
    path = Path('data')
    df = pd.read_csv(path/'train.csv')
    ds = CData(df, path/'train_images')
    print(ds[0])
