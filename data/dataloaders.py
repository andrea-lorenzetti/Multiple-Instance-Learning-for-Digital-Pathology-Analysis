from torch.utils.data import DataLoader
from typing import List, Tuple
import torch

def ls_mil_collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[List[torch.Tensor], torch.Tensor]:
    bags, labels = zip(*batch)
    return list(bags), torch.stack(labels)

def mask_mil_collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bags, labels = zip(*batch)
    max_inst = max([b.size(0) for b in bags])
    pad_bags = [torch.cat([b, torch.zeros(max_inst - b.size(0), *b.shape[1:])], dim=0) for b in bags]
    masks = [torch.cat([torch.ones(b.size(0)), torch.zeros(max_inst - b.size(0))], dim=0).bool() for b in bags]
    return torch.stack(pad_bags), torch.stack(masks), torch.stack(labels)

class MILDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, collate_fn=ls_mil_collate_fn)

class MILMaskDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, collate_fn=mask_mil_collate_fn)
