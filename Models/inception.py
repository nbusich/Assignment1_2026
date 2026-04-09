import torch
import torch.nn as nn
from .conv import Conv1d


class InceptionBlock(nn.Module):
    """
    1D Inception module with parallel branches of different kernel sizes.
    If kernels is an int k, build branches with odd kernel sizes up to k:
    e.g. k=7 -> (1, 3, 5, 7)
    """

    def __init__(self, in_channels, out_channels, kernels=(1, 3, 5, 7), init_name="kaiming"):
        super().__init__()

        if isinstance(kernels, int):
            assert kernels >= 1 and kernels % 2 == 1, (
                f"integer kernels must be a positive odd number, got {kernels}"
            )
            kernels = tuple(range(1, kernels + 1, 2))

        assert out_channels % len(kernels) == 0, (
            f"out_channels ({out_channels}) must be divisible by number of branches ({len(kernels)})"
        )
        branch_ch = out_channels // len(kernels)

        self.branches = nn.ModuleList()
        for k in kernels:
            branch = nn.Sequential(
                Conv1d(in_channels, branch_ch, kernel_size=1, padding=0, init_name=init_name),
                Conv1d(branch_ch, branch_ch, kernel_size=k, padding=k // 2, init_name=init_name),
            )
            self.branches.append(branch)

    def forward(self, x):
        return torch.cat([b(x) for b in self.branches], dim=1)