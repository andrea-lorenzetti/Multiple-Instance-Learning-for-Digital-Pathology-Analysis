import os
import os.path as osp
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import torch

class MILDataset(Dataset):
    """Dataset for Multiple Instance Learning (MIL)."""
    def __init__(self, root: str, transform=None):
        self.root = root
        self.labels = pd.read_csv(osp.join(root, "labels.csv"))
        self.transform = transform if transform else ToTensor()

    def __len__(self):
        return len(self.labels)

    def _read_instance(self, wsi, patch):
        return Image.open(osp.join(self.root, "bags", wsi, patch)).convert("RGB")

    def __getitem__(self, idx):
        wsi, label = self.labels.iloc[idx]
        patches = [self._read_instance(wsi, patch) for patch in os.listdir(osp.join(self.root, "bags", wsi))]
        if self.transform:
            patches = [self.transform(patch) for patch in patches]
        bag = torch.stack(patches, dim=0)
        lbl = torch.tensor(label)
        return bag, lbl
