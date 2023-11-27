import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class NoAug(nn.Module):
    def forward(self, x):
        return x

class LIXAug(nn.Module):
    def __init__(self, s):
        super().__init__()
        self.s = s

    def forward(self, x):
        h_shift = np.random.rand() * 2 * self.s - self.s
        w_shift = np.random.rand() * 2 * self.s - self.s
        _, _, h, w = x.size()
        h_indices, w_indices = torch.meshgrid(
            torch.arange(h, device=x.device, dtype=torch.int64),
            torch.arange(w, device=x.device, dtype=torch.int64)
        )
        shifted_h_indices = torch.clamp(h_indices + h_shift,min=0,max=h-1)
        shifted_w_indices = torch.clamp(w_indices + w_shift,min=0,max=w-1)
        floored_shifted_h_indices = torch.floor(shifted_h_indices).long()
        ceiled_shifted_h_indices = torch.ceil(shifted_h_indices).long()
        floored_shifted_w_indices = torch.floor(shifted_w_indices).long()
        ceiled_shifted_w_indices = torch.ceil(shifted_w_indices).long()
        result = x.clone()
        result[:, :, h_indices, w_indices] = (
            x[:, :, floored_shifted_h_indices, floored_shifted_w_indices] * (ceiled_shifted_h_indices - shifted_h_indices) * (ceiled_shifted_w_indices - shifted_w_indices)
            + x[:, :, floored_shifted_h_indices, ceiled_shifted_w_indices] * (ceiled_shifted_h_indices - shifted_h_indices) * (shifted_w_indices - floored_shifted_w_indices)
            + x[:, :, ceiled_shifted_h_indices, floored_shifted_w_indices] * (shifted_h_indices - floored_shifted_h_indices) * (ceiled_shifted_w_indices - shifted_w_indices)
            + x[:, :, ceiled_shifted_h_indices, ceiled_shifted_w_indices] * (shifted_h_indices - floored_shifted_h_indices) * (shifted_w_indices - floored_shifted_w_indices)
        )
        return result

class CombinedRandomShearingAug(nn.Module):
    def __init__(
        self, magnitude_interval_horizontal=(-1, 1), magnitude_interval_vertical=(-1, 1)
    ):
        super().__init__()
        self.magnitude_interval_horizontal = magnitude_interval_horizontal
        self.magnitude_interval_vertical = magnitude_interval_vertical

    def forward(self, x):
        n = x.size(0)
        shear_factor_horizontal = (
            torch.rand(1, device=x.device, dtype=x.dtype).item()
            * (
                self.magnitude_interval_horizontal[1]
                - self.magnitude_interval_horizontal[0]
            )
            + self.magnitude_interval_horizontal[0]
        )
        shear_factor_vertical = (
            torch.rand(1, device=x.device, dtype=x.dtype).item()
            * (
                self.magnitude_interval_vertical[1]
                - self.magnitude_interval_vertical[0]
            )
            + self.magnitude_interval_vertical[0]
        )
        shear_tensor = torch.tensor(
            [[1, shear_factor_horizontal, 0], [shear_factor_vertical, 1, 0]],
            device=x.device,
        )

        sheared = F.affine_grid(
            shear_tensor.repeat(n, 1, 1),
            x.size(),
            align_corners=False,
        )
        return F.grid_sample(x, sheared, align_corners=False, padding_mode="zeros")

# 相似平均 权重动态调整
class RandomWeightedMeanAug(nn.Module):
    def __init__(self, major_weight_range=(0.95, 0.99)):
        super().__init__()
        assert major_weight_range[0] <= major_weight_range[1] and major_weight_range[0] >= 0 and major_weight_range[1] <= 1
        self.major_weight_range = major_weight_range

    def forward(self, x):
        major_weight = (
            torch.rand(1, device=x.device, dtype=x.dtype).item()
            * (
                self.major_weight_range[1]
                - self.major_weight_range[0]
            )
            + self.major_weight_range[0]
        )
        _, c, _, _ = x.size()
        c_indices = torch.arange(c, device=x.device, dtype=torch.int64)
        remaining_weight = (1 - major_weight) / (c - 1)
        result = x.clone()
        remaining_sum = x[:, :, :, :].sum(dim=1) * remaining_weight
        remaining_sum = remaining_sum.reshape(x.shape[0], 1, x.shape[2], x.shape[3]).repeat(1, c, 1, 1)
        result [:, c_indices, :, :] = (
            x[:, c_indices, :, :] * (major_weight - remaining_weight)
            + remaining_sum
        )
        return result

class RandomWhiteGenAug(nn.Module):
    def __init__(self, ratio_interval=(0.05, 0.1), noise_count=1):
        super().__init__()
        assert ratio_interval[0] <= ratio_interval[1] and ratio_interval[0] >= 0 and ratio_interval[1] <= 1
        assert noise_count >= 1
        self.ratio_interval = ratio_interval
        self.noise_count = noise_count

    def forward(self, x):
        _, c, h, w = x.size()
        result = x.clone()
        white_max, _ = torch.max(x[:, :, :, :].view(x.shape[0], c, -1), dim=2, keepdim=True)
        white_max = white_max.view(x.shape[0], c)
        white_noise = white_max * (
            torch.rand(1, device=x.device, dtype=x.dtype).item() * (
                self.ratio_interval[1] - self.ratio_interval[0]
            ) + self.ratio_interval[0]
        )
        # for every channel in noise_map, choose a random position and add white_noise, repeat for noise_count times
        noise_map = torch.zeros_like(x, device=x.device, dtype=x.dtype)
        h_indices = torch.randint(0, h, size=(self.noise_count,), device=x.device, dtype=torch.int64)
        w_indices = torch.randint(0, w, size=(self.noise_count,), device=x.device, dtype=torch.int64)
        noise_map[:, :, h_indices, w_indices] += white_noise.view(x.shape[0], c, 1)
        result += noise_map
        return result
    
class RandomPadResizeAug(nn.Module):
    def __init__(self, pad_interval=(0, 10)):
        super().__init__()
        assert pad_interval[0] <= pad_interval[1] and pad_interval[0] >= 0
        self.pad_interval = pad_interval
        

    def forward(self, x):
        _, _, h, w = x.size()
        # randomly pad left right top bottom of x with zeros
        pad_left = torch.randint(self.pad_interval[0], self.pad_interval[1], size=(1,), device=x.device, dtype=torch.int64).item()
        pad_right = torch.randint(self.pad_interval[0], self.pad_interval[1], size=(1,), device=x.device, dtype=torch.int64).item()
        pad_top = torch.randint(self.pad_interval[0], self.pad_interval[1], size=(1,), device=x.device, dtype=torch.int64).item()
        pad_bottom = torch.randint(self.pad_interval[0], self.pad_interval[1], size=(1,), device=x.device, dtype=torch.int64).item()

        result = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), "constant", 0)

        # resize result to original size
        result = F.interpolate(result, size=(h, w), mode='nearest')

        return result
    
class RandomWhiteGenAugEnhanced(nn.Module):
    def __init__(self, ratio_interval=(0.05, 0.1), noise_count_interval=(1,5), h_margins=(5, 5), w_margins=(5, 5)):
        super().__init__()
        assert ratio_interval[0] <= ratio_interval[1] and ratio_interval[0] >= 0 and ratio_interval[1] <= 1
        assert noise_count_interval[0] <= noise_count_interval[1] and noise_count_interval[0] >= 0
        assert h_margins[0] > 0 and h_margins[1] > 0 and w_margins[0] > 0 and w_margins[1] > 0
        self.ratio_interval = ratio_interval
        self.noise_count_interval = noise_count_interval
        self.h_margins = h_margins
        self.w_margins = w_margins

    def forward(self, x):
        _, c, h, w = x.size()
        result = x.clone()
        white_max, _ = torch.max(x[:, :, :, :].view(x.shape[0], c, -1), dim=2, keepdim=True)
        white_max = white_max.view(x.shape[0], c)
        white_noise = white_max * (
            torch.rand(1, device=x.device, dtype=x.dtype).item() * (
                self.ratio_interval[1] - self.ratio_interval[0]
            ) + self.ratio_interval[0]
        )
        # for every channel in noise_map, choose a random position and add white_noise, repeat for noise_count times
        noise_count = torch.randint(self.noise_count_interval[0], self.noise_count_interval[1], size=(1,), device=x.device, dtype=torch.int64).item()
        noise_map = torch.zeros_like(x, device=x.device, dtype=x.dtype)
        h_interval = [(0, self.h_margins[0]), (h - self.h_margins[1], h)]
        h_interval = h_interval[torch.randint(0, 2, size=(1,), device=x.device, dtype=torch.int64).item()]
        w_interval = [(0, self.w_margins[0]), (w - self.w_margins[1], w)]
        w_interval = w_interval[torch.randint(0, 2, size=(1,), device=x.device, dtype=torch.int64).item()]
        h_indices = torch.randint(h_interval[0], h_interval[1], size=(noise_count,), device=x.device, dtype=torch.int64)
        w_indices = torch.randint(w_interval[0], w_interval[1], size=(noise_count,), device=x.device, dtype=torch.int64)
        noise_map[:, :, h_indices, w_indices] += white_noise.view(x.shape[0], c, 1)
        result += noise_map
        return result

feat_augmentations = [NoAug(), LIXAug(1), CombinedRandomShearingAug((-1 / 3, 1 / 3), (-1 / 3, 1 / 3)), RandomWeightedMeanAug((0.95,0.99)), RandomWhiteGenAug(ratio_interval=(0.2,0.3), noise_count=3), RandomPadResizeAug(pad_interval=(0,10)), RandomWhiteGenAugEnhanced(),]


class FeatAug(nn.Module):
    def __init__(self, aug_type, c=32, h=35, w=35):
        super().__init__()
        self.aug = feat_augmentations[aug_type - 1]
        self.c = c
        self.h = h
        self.w = w

    def forward(self, x):
        ori_feat_map = x.reshape(-1, self.c, self.h, self.w)
        aug_feat_map = self.aug(ori_feat_map)
        return aug_feat_map.view(x.shape[0], -1)
