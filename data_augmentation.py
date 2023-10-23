# Implementations for different image transformation used for data augmentation
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from kornia.color.hsv import rgb_to_hsv, hsv_to_rgb
import cv2
import math


class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, "replicate")
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(
            -1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype
        )[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(
            0, 2 * self.pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype
        )
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)


class RandomShearingAug(nn.Module):
    def __init__(self, direction="horizontal", magnitude_interval=(-1, 1)):
        super().__init__()
        self.direction = direction
        self.magnitude_interval = magnitude_interval

    def forward(self, x):
        n = x.size(0)
        shear_factor = (
            torch.rand(1, device=x.device, dtype=x.dtype).item()
            * (self.magnitude_interval[1] - self.magnitude_interval[0])
            + self.magnitude_interval[0]
        )
        shear_tensor = (
            torch.tensor([[1, shear_factor, 0], [0, 1, 0]], device=x.device)
            if self.direction == "horizontal"
            else torch.tensor([[1, 0, 0], [shear_factor, 1, 0]], device=x.device)
        )

        sheared = F.affine_grid(
            shear_tensor.repeat(n, 1, 1),
            x.size(),
            align_corners=False,
        )
        return F.grid_sample(x, sheared, align_corners=False, padding_mode="zeros")


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


class RandomNoiseInjectionAug(nn.Module):
    def __init__(self, magnitude_interval=(0, 0.1), float_tensor=True):
        super().__init__()
        self.magnitude_interval = magnitude_interval
        self.float_tensor = float_tensor

    def forward(self, x):
        noise = (
            (
                torch.rand_like(x, device=x.device, dtype=x.dtype)
                * (self.magnitude_interval[1] - self.magnitude_interval[0])
                + self.magnitude_interval[0]
            )
            if self.float_tensor
            else (
                torch.randint_like(
                    x, self.magnitude_interval[0], self.magnitude_interval[1]
                )
            )
        )
        return (x + noise).clamp(0, 1 if self.float_tensor else 255).to(x.dtype)


class RandomColorShiftingAug(nn.Module):
    def __init__(self, magnitude_interval=(0, 0.1), float_tensor=True):
        super().__init__()
        self.magnitude_interval = magnitude_interval
        self.float_tensor = float_tensor

    def forward(self, x):
        shift = (
            (
                torch.rand(1, device=x.device, dtype=x.dtype)
                * (self.magnitude_interval[1] - self.magnitude_interval[0])
                + self.magnitude_interval[0]
            ).item()
            if self.float_tensor
            else (
                torch.randint(
                    self.magnitude_interval[0], self.magnitude_interval[1], (1,)
                )
            ).item()
        )
        return (x + shift).clamp(0, 1 if self.float_tensor else 255).to(x.dtype)


class RandomColorScalingAug(nn.Module):
    def __init__(self, magnitude_interval=(0, 1), float_tensor=True):
        super().__init__()
        self.magnitude_interval = magnitude_interval
        self.float_tensor = float_tensor

    def forward(self, x):
        scale = (
            torch.rand(1, device=x.device, dtype=x.dtype)
            * (self.magnitude_interval[1] - self.magnitude_interval[0])
            + self.magnitude_interval[0]
        ).item()
        result = (x * scale).clamp(0, 1 if self.float_tensor else 255)
        if self.float_tensor:
            return result
        else:
            return result.round().to(x.dtype)


class ColorInversionAug(nn.Module):
    def __init__(self, float_tensor=True):
        super().__init__()
        self.float_tensor = float_tensor

    def forward(self, x):
        color_max = 1 if self.float_tensor else 255
        return (color_max - x).to(x.dtype)


import math


class DRQRandomHueShiftingAug(nn.Module):
    def __init__(self, magnitude_interval=(0, 0.1), float_tensor=True):
        super().__init__()
        self.magnitude_interval = magnitude_interval
        self.float_tensor = float_tensor

    def single_forward(self, x, shift):
        hsv = rgb_to_hsv(x) if self.float_tensor else rgb_to_hsv(x / 255)
        hsv[:, 0, :, :] = (hsv[:, 0, :, :] + shift) % (2 * math.pi)
        return (
            hsv_to_rgb(hsv) if self.float_tensor else (hsv_to_rgb(hsv) * 255).round()
        ).to(x.dtype)

    def forward(self, x):
        shift = (
            torch.rand(1, device=x.device, dtype=x.dtype)
            * (self.magnitude_interval[1] - self.magnitude_interval[0])
            + self.magnitude_interval[0]
        ).item()
        transformed_imgs = (
            self.single_forward(x[:, 0:3, :, :], shift),
            self.single_forward(x[:, 3:6, :, :], shift),
            self.single_forward(x[:, 6:9, :, :], shift),
        )
        return torch.cat(transformed_imgs, dim=1)


class DRQRandomSaturationScalingAug(nn.Module):
    def __init__(self, magnitude_interval=(0, 0.1), float_tensor=True):
        super().__init__()
        self.magnitude_interval = magnitude_interval
        self.float_tensor = float_tensor

    def single_forward(self, x, scale):
        hsv = rgb_to_hsv(x) if self.float_tensor else rgb_to_hsv(x / 255)
        hsv[:, 1, :, :] = (hsv[:, 1, :, :] * scale) % 1
        return (
            hsv_to_rgb(hsv) if self.float_tensor else (hsv_to_rgb(hsv) * 255).round()
        ).to(x.dtype)

    def forward(self, x):
        scale = (
            torch.rand(1, device=x.device, dtype=x.dtype)
            * (self.magnitude_interval[1] - self.magnitude_interval[0])
            + self.magnitude_interval[0]
        ).item()
        transformed_imgs = (
            self.single_forward(x[:, 0:3, :, :], scale),
            self.single_forward(x[:, 3:6, :, :], scale),
            self.single_forward(x[:, 6:9, :, :], scale),
        )
        return torch.cat(transformed_imgs, dim=1)


class KernelAug(nn.Module):
    def __init__(self, kernel, float_tensor=True):
        super().__init__()
        self.kernel = kernel
        self.float_tensor = float_tensor

    def forward(self, x):
        kernel = self.kernel.to(x.device).to(x.dtype)
        c = x.size(1)
        conved = F.conv2d(
            x, kernel.repeat(c, 1, 1, 1), padding=kernel.size(0) // 2, groups=c
        )
        conved = conved if self.float_tensor else conved.round()
        return conved.clamp(0, 1 if self.float_tensor else 255).to(x.dtype)


class GaussianBlurAug(KernelAug):
    def __init__(self, kernel_size=3, sigma=1, float_tensor=True):
        kernel = torch.tensor(
            cv2.getGaussianKernel(kernel_size, sigma),
            dtype=torch.float32,
        )
        kernel = kernel @ kernel.t()
        super().__init__(kernel, float_tensor)


class DRQRandomCutOutAug(nn.Module):
    def __init__(self, region_size=(2, 2), fill_color=(1, 1, 1), float_tensor=True):
        super().__init__()
        self.region_size = region_size
        self.fill_color = fill_color
        self.float_tensor = float_tensor

    def single_forward(self, x):
        _, c, h, w = x.size()
        if self.float_tensor:
            fill_color = torch.tensor(self.fill_color, device=x.device, dtype=x.dtype)
        else:
            fill_color = (
                torch.tensor(self.fill_color, device=x.device, dtype=x.dtype) * 255
            )
        region_size = torch.tensor(self.region_size, device=x.device, dtype=torch.int)

        x1 = torch.randint(0, h - region_size[0] + 1, (1,)).item()
        y1 = torch.randint(0, w - region_size[1] + 1, (1,)).item()
        x2 = x1 + region_size[0]
        y2 = y1 + region_size[1]
        result = torch.clone(x)
        for channel in range(c):
            result[:, channel, x1:x2, y1:y2] = fill_color[channel]
        return result

    def forward(self, x):
        transformed_imgs = (
            self.single_forward(x[:, 0:3, :, :]),
            self.single_forward(x[:, 3:6, :, :]),
            self.single_forward(x[:, 6:9, :, :]),
        )
        return torch.cat(transformed_imgs, dim=1)


# identifier c
augmentations = [
    RandomShiftsAug(4),
    CombinedRandomShearingAug((-1 / 3, 1 / 3), (-1 / 3, 1 / 3)),
]


class DataAug(nn.Module):
    def __init__(self, da_type):
        super().__init__()
        self.data_aug_type = da_type
        self.aug = augmentations[da_type - 1]

    def forward(self, x):
        return self.aug(x)
