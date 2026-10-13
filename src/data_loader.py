import os 
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image

CLASSES =  ['neutral','happiness','surprise','sadness','anger','disgust','fear','contempt','unknown','NF']


class FerDataset(Dataset):
    def __init__(self, root, transform = None):
        self.root = Path(root)
        self.transform = transform

        self._samples, self._labels = self._collect_samples()
        
        
    def __getitem__(self, index):
        path = self.root / self._samples[index]
        label = self._labels[index]
        
        img = Image.open(path)

        img = img.convert("RGB") if img.mode != "RGB" else img
        if self.transform:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self._samples)

    def _collect_samples(self):
        label_path = self.root / 'label.csv'
        
        data = pd.read_csv(label_path, names=['Image name','Box'] + CLASSES)
        
        samples = data["Image name"].values
        labels = data[CLASSES].to_numpy(dtype=float)
        labels = labels / labels.sum(axis=1, keepdims=True)

        return samples, labels


if __name__ == "__main__":
    a = FerDataset(root = "data/test/")
    idx = 100
    print(len(a))
    labels = a[idx][1]
    for cls, value in zip(CLASSES, labels):
        if value > 0:
            print(f"{cls}: {value}")
    a[idx][0].show()

