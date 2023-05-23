import math
import warnings
from typing import Optional, List

import numpy as np
import torch

class ColorUtils:
    _hex_colors_orig = np.array(['#2c3e50', '#e74c3c', '#27ae60', '#3498db', '#CDDC39', '#f39c12', '#795548', '#8e44ad',
                           '#3F51B5', '#7f8c8d', '#e84393', '#607D8B', '#8e44ad', '#009688'])
    hex_colors = _hex_colors_orig

    rgb_colors = torch.tensor([
                [244, 67, 54],
                [156, 39, 176],
                [63, 81, 181],
                [3, 169, 244],
                [0, 150, 136],
                [139, 195, 74],
                [255, 235, 59],
                [255, 152, 0],
                [121, 85, 72],
                [96, 125, 139],
                [233, 30, 99],
                [103, 58, 183],
                [33, 150, 243],
                [0, 188, 212],
                [76, 175, 80],
                [205, 220, 57],
                [255, 193, 7],
                [255, 87, 34],
                [158, 158, 158]], dtype=torch.float)

    _rgb_colors_orig = rgb_colors

    rgb_feature_colors = rgb_colors
    feature_labels: Optional[List[str]] = None

    @staticmethod
    def rgb2hex(r: int, g: int, b: int):
        return f'#{r:02x}{g:02x}{b:02x}'

    @staticmethod
    def rgb2hex_tensor(ten: torch.Tensor):
        ten = torch.round(ten).to(int)
        return ColorUtils.rgb2hex(ten[0].item(), ten[1].item(), ten[2].item())

    @classmethod
    def ensure_min_hex_colors(cls, required_colors: int):
        if cls.hex_colors.shape[0] < required_colors:
            new_num_colors = math.ceil(required_colors / cls._hex_colors_orig.shape[0])
            warnings.warn(
                f"Only {cls.hex_colors.shape[0]} colors but {required_colors} needed! Repeating original "
                f"{new_num_colors} times.")
            cls.hex_colors = np.tile(cls._hex_colors_orig, new_num_colors)

    @classmethod
    def ensure_min_rgb_colors(cls, required_colors: int | torch.Tensor):
        if cls.rgb_colors.shape[0] < required_colors:
            new_num_colors = math.ceil(required_colors / cls._rgb_colors_orig.shape[0])
            warnings.warn(
                f"Only {cls.rgb_colors.shape[0]} colors given to distinguish {required_colors} "
                f"cluster! Repeating original {new_num_colors} times.")
            cls.rgb_colors = cls._rgb_colors_orig.repeat(new_num_colors, 1)

    @classmethod
    def ensure_min_rgb_feature_colors(cls, required_colors: int | torch.Tensor):
        if cls.rgb_colors.shape[0] < required_colors:
            new_num_colors = math.ceil((required_colors - cls.rgb_feature_colors.shape[0]) /
                                       cls._rgb_colors_orig.shape[0])
            warnings.warn(
                f"Only {cls.rgb_colors.shape[0]} colors given to distinguish {required_colors} "
                f"features! Adding {new_num_colors * cls._rgb_colors_orig.shape[0]} from rgb colors.")
            cls.rgb_colors = torch.cat((cls.rgb_feature_colors, cls._rgb_colors_orig.repeat(new_num_colors, 1)), dim=0)