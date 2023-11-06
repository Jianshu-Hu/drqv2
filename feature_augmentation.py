import torch.nn as nn

class NoAug(nn.Module):
    def forward(self, x):
        return x

feat_augmentations = [NoAug()]

class FeatAug(nn.Module):
    def __init__(self, aug_type):
        super().__init__()
        self.aug = feat_augmentations[aug_type - 1]

    def forward(self, x):
        return self.aug(x)