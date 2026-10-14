from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

CLASSES  = ['neutral','happiness','surprise','sadness','anger','disgust','fear','contempt','unknown','NF']
EMOTIONS = ['neutral','happiness','surprise','sadness','anger','disgust','fear','contempt']

class FerDataset(Dataset):
    def __init__(self, root, transform=None, excluded_emotions=None):
        self.root = Path(root)
        self.transform = transform
        self.excluded_emotions = list(excluded_emotions or [])

        invalid = set(self.excluded_emotions) - set(EMOTIONS)
        if invalid:
            raise ValueError(f"Unknown emotions: {sorted(invalid)}")

        self.emotions = [e for e in EMOTIONS if e not in self.excluded_emotions]
        self._samples, self._labels = self._collect_samples()

    def __getitem__(self, index):
        path = self.root / self._samples[index]
        with Image.open(path) as img:
            img = img.convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self._labels[index]

    def __len__(self):
        return len(self._samples)

    def _collect_samples(self):
        data = pd.read_csv(self.root / "label.csv", names=["Image name", "Box"] + CLASSES)

        data = data[(data["NF"] <= 5) & (data["unknown"] <= 5)]

        votes = data[EMOTIONS].to_numpy(dtype=float)
        dominant = votes.argmax(axis=1)

        excluded_idx = [EMOTIONS.index(e) for e in self.excluded_emotions]
        kept_idx = [EMOTIONS.index(e) for e in self.emotions]

        keep = (votes.sum(axis=1) > 0) & ~np.isin(dominant, excluded_idx)

        samples = data["Image name"].values[keep]
        kept_votes = votes[keep][:, kept_idx]
        labels = (kept_votes / kept_votes.sum(axis=1, keepdims=True)).astype("float32")
        return samples, labels


if __name__ == "__main__":
    a = FerDataset("data/train/", excluded_emotions=["contempt"])
    b = FerDataset("data/train/", excluded_emotions=[e for e in EMOTIONS if e != "contempt"])
    c = FerDataset("data/train/")

    print(len(a), len(b), len(c))
    assert len(a) + len(b) == len(c)

    idx = 99
    for cls, value in zip(b.emotions, b[idx][1]):
        if value > 0:
            print(f"{cls}: {value}")
    b[idx][0].show()
    