import torch
import torch.nn as nn

def conv_block(in_ch: int, out_ch: int, norm: bool = True) -> nn.Sequential:
    layers = [nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=not norm)]
    if norm:
        layers.append(nn.BatchNorm2d(out_ch))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)

def deconv_block(in_ch: int, out_ch: int, dropout: bool = False) -> nn.Sequential:
    layers = [nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False),
              nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)]
    if dropout:
        layers.append(nn.Dropout(0.5))
    return nn.Sequential(*layers)

class MultiTaskUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        base_channels: int = 32,
        semantic_out: int = 40,
        depth_out: int = 1,
        obstacle_out: int = 1
    ) -> None:
        super().__init__()
        self.base_ch = base_channels
        # 下采样（输入256×256）
        self.down1 = conv_block(in_channels, base_channels, norm=False)  # 256→128
        self.down2 = conv_block(base_channels, base_channels*2)         # 128→64
        self.down3 = conv_block(base_channels*2, base_channels*4)      # 64→32
        self.down4 = conv_block(base_channels*4, base_channels*8)       # 32→16
        self.down5 = conv_block(base_channels*8, base_channels*8)       # 16→8
        self.down6 = conv_block(base_channels*8, base_channels*8)       # 8→4
        self.down7 = conv_block(base_channels*8, base_channels*8)       # 4→2
        self.down8 = conv_block(base_channels*8, base_channels*8)       # 2→1

        # 上采样（新增1层，最终恢复到256×256）
        self.up1 = deconv_block(base_channels*8, base_channels*8, dropout=True) # 1→2
        self.up2 = deconv_block(base_channels*16, base_channels*8, dropout=True)# 2→4
        self.up3 = deconv_block(base_channels*16, base_channels*8, dropout=True)# 4→8
        self.up4 = deconv_block(base_channels*16, base_channels*8)               # 8→16
        self.up5 = deconv_block(base_channels*16, base_channels*4)               # 16→32
        self.up6 = deconv_block(base_channels*8, base_channels*2)                # 32→64
        self.up7 = deconv_block(base_channels*4, base_channels)                   # 64→128
        self.up8 = deconv_block(base_channels*2, base_channels)                   # 128→256 【新增层】

        # 输出头（输入256×256，输出和标签同尺寸）
        self.semantic_head = nn.Conv2d(base_channels, semantic_out, 3, 1, 1)  
        self.depth_head = nn.Conv2d(base_channels, depth_out, 3, 1, 1)        
        self.obstacle_head = nn.Conv2d(base_channels, obstacle_out, 3, 1, 1)  

        self.softmax = nn.Softmax(dim=1)  
        self.tanh = nn.Tanh()             
        self.sigmoid = nn.Sigmoid()       

    def forward(self, x: torch.Tensor) -> dict:
        # 下采样
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)

        # 上采样+跳跃连接
        u1 = self.up1(d8)
        u2 = self.up2(torch.cat([u1, d7], dim=1))
        u3 = self.up3(torch.cat([u2, d6], dim=1))
        u4 = self.up4(torch.cat([u3, d5], dim=1))
        u5 = self.up5(torch.cat([u4, d4], dim=1))
        u6 = self.up6(torch.cat([u5, d3], dim=1))
        u7 = self.up7(torch.cat([u6, d2], dim=1))
        u8 = self.up8(torch.cat([u7, d1], dim=1)) # 最终输出256×256

        # 任务输出
        semantic = self.softmax(self.semantic_head(u8))
        depth = self.tanh(self.depth_head(u8))
        obstacle = self.sigmoid(self.obstacle_head(u8))

        return {
            "semantic": semantic,
            "depth": depth,
            "obstacle": obstacle
        }