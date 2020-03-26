"""
Boilerplate data loaders. Use these or easily add your own.
"""

import torch

import numpy as np

from torchvision import datasets, transforms


class MNISTDataLoader(torch.utils.data.dataloader.DataLoader):
    """
    MNIST data loading demo using BaseDataLoader

    Config should be structured
        "data_loader": {
        "type": "MnistDataLoader",
        "args":
            "data_dir": "data/MNIST"
            "batch_size": 8000,
            "num_workers": 2,
            "percent_used_for_validation", 0.1
        }
    """
    def __init__(self,
                 data_dir="",
                 batch_size=16,
                 num_workers=1,
                 stage="training",
                 percent_used_for_validation=0.9):
        assert stage in ["training", "validation", "test"]
        trsfm = transforms.Compose([
            transforms.ToTensor()
            ])

        train = True
        if stage == "test":
            train = False

        self.dataset = datasets.MNIST(data_dir, train=train, download=True, transform=trsfm)

        # Perform validation set split unless this is test set. Note this assumes dataset is pre-randomized.
        indices = np.arange(len(self.dataset))
        if train:
            split_idx = int(np.floor(len(self.dataset) * (1 - percent_used_for_validation)))
            if stage == "training":
                indices = indices[:split_idx]
            elif stage == "validation":
                indices = indices[split_idx:]
        sampler = torch.utils.data.sampler.SubsetRandomSampler(indices)
        super(MNISTDataLoader, self).__init__(self.dataset,
                                              sampler=sampler,
                                              batch_size=batch_size,
                                              num_workers=num_workers,
                                              drop_last=True)
        # This data has shape [batch_size, 1, 28, 28]


class FashionMNISTDataLoader(torch.utils.data.dataloader.DataLoader):
    """
    FashionMNIST data loading demo using BaseDataLoader

    Config should be structured
        "data_loader": {
        "type": "FashionMnistDataLoader",
        "args":
            "root": "data/FashionMNIST"
            "batch_size": 8000,
            "shuffle": true,
            "num_workers": 2,
            "percent_used_for_validation", 0.9
        }
    """

    def __init__(self,
                 root="",
                 batch_size=16,
                 shuffle=True,
                 num_workers=1,
                 stage="training",
                 percent_used_for_validation=0.9):
        assert stage in ["training", "validation", "test"]
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2861,), (0.3530,))

            ])

        train = True
        if stage == "test":
            train = False

        self.dataset = datasets.FashionMNIST(root, train=train, download=True, transform=trsfm)

        # Perform validation set split unless this is test set. Note this assumes dataset is pre-randomized.
        indices = np.arange(len(self.dataset))
        if train:
            split_idx = int(np.floor(len(self.dataset) * (1 - percent_used_for_validation)))
            if stage == "training":
                indices = indices[:split_idx]
            elif stage == "validation":
                indices = indices[split_idx:]
        sampler = torch.utils.data.sampler.SubsetRandomSampler(indices)
        super(MNISTDataLoader, self).__init__(self.dataset,
                                              sampler=sampler,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              drop_last=True)
        # This data has shape [batch_size, 1, 28, 28]
