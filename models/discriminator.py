import torch
import torch.nn as nn


def d_block(in_ch: int, out_ch: int, stride: int = 2, norm: bool = True) -> nn.Sequential:
    layers = [nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=stride, padding=1, bias=not norm)]
    if norm:
        layers.append(nn.BatchNorm2d(out_ch))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)


class PatchDiscriminator(nn.Module):
    
    def __init__(self, in_channels: int = 4, base_channels: int = 64) -> None:
        super().__init__()
        self.model = nn.Sequential(
            d_block(in_channels, base_channels, norm=False),            # 128
            d_block(base_channels, base_channels * 2),                  # 64
            d_block(base_channels * 2, base_channels * 4),              # 32
            d_block(base_channels * 4, base_channels * 8, stride=1),    # 31
            nn.Conv2d(base_channels * 8, 1, kernel_size=4, stride=1, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)