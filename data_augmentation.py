# Implementations for different image transformation used for data augmentation
import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia
import numpy as np

class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x,
                             grid,
                             padding_mode='zeros',
                             align_corners=False)

class GridMask():
    def __init__(self, grid_m=8, grid_n=8) -> None:
        self.m=grid_m
        self.n=grid_n
    
    def __call__(self, inputs):
        inputs = inputs.clone()
        for k, input in enumerate(inputs):
            h, w = input.shape[1:]  # input.shape = (3, 32, 32)
            cell_h = h // self.m
            cell_w = w // self.n
            for i in range(self.m):
                for j in range(self.n):
                    if (i+j)%2==0:
                        inputs[k][:, i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w] = 0
        return inputs

class HideandSeek(GridMask):
    def __init__(self, grid_m=8, grid_n=8, p=0.4) -> None:
        super().__init__(grid_m, grid_n)
        self.p=p
    
    def __call__(self, inputs):
        inputs = inputs.clone()
        for k, input in enumerate(inputs):
            h, w = input.shape[1:]  # input.shape = (3, 32, 32)
            cell_h = h // self.m
            cell_w = w // self.n
            for i in range(self.m):
                for j in range(self.n):
                    prob=np.random.rand()
                    if (i+j)%2==0 and prob<self.p:
                        inputs[k][:, i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w] = 0
        return inputs

class DataAug(nn.Module):
    def __init__(self, da_type):
        super().__init__()
        self.data_aug_type = da_type
        if self.data_aug_type == 1:
            # random shift
            self.aug = RandomShiftsAug(4)
        elif self.data_aug_type == 2:
            self.aug = kornia.augmentation.RandomRotation(degrees=180., p=1., same_on_batch=False)
        elif self.data_aug_type == 3:
            self.aug = HideandSeek(grid_m=8, grid_n=8, p=0.4)

    def forward(self, x):
        return self.aug(x)