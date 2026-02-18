"""Central project configuration values."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    input_dim: tuple[int, int, int] = (64, 64, 3)
    conv_filters: tuple[int, ...] = (32, 64, 64, 128)
    conv_kernel_sizes: tuple[int, ...] = (4, 4, 4, 4)
    conv_strides: tuple[int, ...] = (2, 2, 2, 2)
    z_dim: int = 32
    epochs: int = 10
    batch_size: int = 32
