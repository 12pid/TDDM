import torch
import torch.nn as nn
import torch.nn.functional as F


def drop_block_2d(
        x, drop_prob: float = 0.1, block_size: int = 7,  gamma_scale: float = 1.0,
        with_noise: bool = False, inplace: bool = False, batchwise: bool = False):

    B, C, H, W = x.shape
    total_size = W * H
    clipped_block_size = min(block_size, min(W, H))
    # seed_drop_rate, the gamma parameter
    gamma = gamma_scale * drop_prob * total_size / clipped_block_size ** 2 / (
        (W - block_size + 1) * (H - block_size + 1))

    # Forces the block to be inside the feature map.
    w_i, h_i = torch.meshgrid(torch.arange(W).to(x.device), torch.arange(H).to(x.device))
    valid_block = ((w_i >= clipped_block_size // 2) & (w_i < W - (clipped_block_size - 1) // 2)) & \
                  ((h_i >= clipped_block_size // 2) & (h_i < H - (clipped_block_size - 1) // 2))
    valid_block = torch.reshape(valid_block, (1, 1, H, W)).to(dtype=x.dtype)

    if batchwise:
        # one mask for whole batch, quite a bit faster
        uniform_noise = torch.rand((1, C, H, W), dtype=x.dtype, device=x.device)
    else:
        uniform_noise = torch.rand_like(x)
    block_mask = ((2 - gamma - valid_block + uniform_noise) >= 1).to(dtype=x.dtype)
    block_mask = -F.max_pool2d(
        -block_mask,
        kernel_size=clipped_block_size,  # block_size,
        stride=1,
        padding=clipped_block_size // 2)

    if with_noise:
        normal_noise = torch.randn((1, C, H, W), dtype=x.dtype, device=x.device) if batchwise else torch.randn_like(x)
        if inplace:
            x.mul_(block_mask).add_(normal_noise * (1 - block_mask))
        else:
            x = x * block_mask + normal_noise * (1 - block_mask)
    else:
        normalize_scale = (block_mask.numel() / block_mask.to(dtype=torch.float32).sum().add(1e-7)).to(x.dtype)
        if inplace:
            x.mul_(block_mask * normalize_scale)
        else:
            x = x * block_mask * normalize_scale
    return x


def drop_block_fast_2d(
        x: torch.Tensor, drop_prob: float = 0.1, block_size: int = 7,
        gamma_scale: float = 1.0, with_noise: bool = False, inplace: bool = False, batchwise: bool = False):

    B, C, H, W = x.shape
    total_size = W * H
    clipped_block_size = min(block_size, min(W, H))
    gamma = gamma_scale * drop_prob * total_size / clipped_block_size ** 2 / (
            (W - block_size + 1) * (H - block_size + 1))

    if batchwise:
        # one mask for whole batch, quite a bit faster
        block_mask = torch.rand((1, C, H, W), dtype=x.dtype, device=x.device) < gamma
    else:
        # mask per batch element
        block_mask = torch.rand_like(x) < gamma
    block_mask = F.max_pool2d(
        block_mask.to(x.dtype), kernel_size=clipped_block_size, stride=1, padding=clipped_block_size // 2)

    if with_noise:
        normal_noise = torch.randn((1, C, H, W), dtype=x.dtype, device=x.device) if batchwise else torch.randn_like(x)
        if inplace:
            x.mul_(1. - block_mask).add_(normal_noise * block_mask)
        else:
            x = x * (1. - block_mask) + normal_noise * block_mask
    else:
        block_mask = 1 - block_mask
        normalize_scale = (block_mask.numel() / block_mask.to(dtype=torch.float32).sum().add(1e-7)).to(dtype=x.dtype)
        if inplace:
            x.mul_(block_mask * normalize_scale)
        else:
            x = x * block_mask * normalize_scale
    return x


class DropBlock2d(nn.Module):

    def __init__(self,
                 drop_prob=0.1,
                 block_size=7,
                 gamma_scale=1.0,
                 with_noise=False,
                 inplace=False,
                 batchwise=False,
                 fast=True):
        super(DropBlock2d, self).__init__()
        self.drop_prob = drop_prob
        self.gamma_scale = gamma_scale
        self.block_size = block_size
        self.with_noise = with_noise
        self.inplace = inplace
        self.batchwise = batchwise
        self.fast = fast

    def forward(self, x):
        if not self.training or not self.drop_prob:
            return x
        if self.fast:
            return drop_block_fast_2d(
                x, self.drop_prob, self.block_size, self.gamma_scale, self.with_noise, self.inplace, self.batchwise)
        else:
            return drop_block_2d(
                x, self.drop_prob, self.block_size, self.gamma_scale, self.with_noise, self.inplace, self.batchwise)


def drop_path(x, drop_prob: float = 0., training: bool = False):

    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
