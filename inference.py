# evaluate the test data
from torch.utils.data import Dataset, DataLoader
import os
import torch
import pandas as pd
from glob import glob
from PIL import Image
from model import ImageClassifier
import albumentations as A
import cv2


TARGET_SIZE = 224


class config:
    batch_size = 8
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # valid_tfms = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(
    # ), transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    valid_tfms = A.Compose([
        A.SmallestMaxSize(max_size=TARGET_SIZE, p=1.0),
        A.CenterCrop(height=TARGET_SIZE, width=TARGET_SIZE, p=1.0),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        A.ToTensorV2(),
    ])

    def predict_batch(model, test_dl):
        preds = []
        with torch.no_grad():
            for i, (img, label) in enumerate(test_dl):
                img = img.to(config.device)
                output = model(img)
                pred = torch.argmax(output, dim=1).numpy().tolist()
                preds.extend(pred)
        return preds


class LeafDataset(Dataset):
    def __init__(self, data_dir, transforms=None):
        self.images = glob(data_dir+"/*")
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        img_path = self.images[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transforms:
            img = self.transforms(image=img)['image']

        return img, torch.tensor(0)


def predict_batch(model, test_dl):
    preds = []
    with torch.no_grad():
        for i, (img, label) in enumerate(test_dl):
            img = img.to(config.device)
            output = model(img)
            pred = torch.argmax(output, dim=1).cpu().numpy().tolist()
            preds.extend(pred)
    return preds


if __name__ == "__main__":
    data_dir = "data"
    test_image_path = os.path.join(data_dir, 'test_images')
    test_dataset = LeafDataset(test_image_path, transforms=config.valid_tfms)
    test_dl = DataLoader(
        test_dataset, batch_size=config.batch_size, shuffle=False)
    CHECKPOINT_PATH = './lightning_logs/version_2/checkpoints/epoch=0-step=5.ckpt'
    model = ImageClassifier.load_from_checkpoint(
        CHECKPOINT_PATH, NUM_CLASSES=5)
    model.eval()
    model.to(config.device)

    preds = predict_batch(model, test_dl)
    # print(preds)

    # submission file
    sample_df = pd.read_csv(os.path.join(data_dir, 'sample_submission.csv'))
    sample_df['image_id'] = os.listdir(test_image_path)
    sample_df['label'] = preds
    df = pd.DataFrame(sample_df)
    df.to_csv('submission.csv', index=False)
    print(df.head())
