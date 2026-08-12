import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Two 3x3 conv + BN + ReLU layers — the basic node block used at every X[i, j]."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class UNetPlusPlus(nn.Module):
    """U-Net++ (Zhou et al., 2018, 'UNet++: A Nested U-Net Architecture for Medical Image
    Segmentation'), as evaluated in Table 1 of the IEW paper (F1 = 79.10%).

    Uses nested, dense skip pathways X[i, j] between the encoder and decoder: each node
    X[i, j] (i = depth level, j = position along the nested path) receives all previous
    nodes at the same depth X[i, 0..j-1] plus an upsampled X[i+1, j-1] from one level
    deeper. This produces a "deep supervision"-friendly grid instead of U-Net's single
    long skip connections, aimed at reducing the semantic gap between encoder and decoder
    features.

    We use 5 depth levels (0..4) and output from X[0, 4] (no deep supervision heads, to
    keep the interface identical to the other single-output models in this repo).
    """

    def __init__(self, in_channels: int = 11, num_classes: int = 6, base_channels: int = 24):
        super().__init__()
        c = base_channels
        # Channel width at each depth level (0 = finest / full resolution)
        ch = [c, c * 2, c * 4, c * 8, c * 16]
        self.depth = 5

        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        # Backbone column (j = 0): standard encoder
        self.blocks = nn.ModuleDict()
        self.blocks["x0_0"] = ConvBlock(in_channels, ch[0])
        self.blocks["x1_0"] = ConvBlock(ch[0], ch[1])
        self.blocks["x2_0"] = ConvBlock(ch[1], ch[2])
        self.blocks["x3_0"] = ConvBlock(ch[2], ch[3])
        self.blocks["x4_0"] = ConvBlock(ch[3], ch[4])

        # Nested skip pathways: X[i, j] for j = 1..(depth-1-i)
        # in_channels for X[i, j] = j*ch[i] (dense concat of X[i,0..j-1]) + ch[i+1] (upsampled)
        for i in range(self.depth - 1):
            for j in range(1, self.depth - i):
                in_c = j * ch[i] + ch[i + 1]
                self.blocks[f"x{i}_{j}"] = ConvBlock(in_c, ch[i])

        self.outc = nn.Conv2d(ch[0], num_classes, kernel_size=1)

    def forward(self, x):
        b = self.blocks
        X = {}

        X["0_0"] = b["x0_0"](x)
        X["1_0"] = b["x1_0"](self.pool(X["0_0"]))
        X["2_0"] = b["x2_0"](self.pool(X["1_0"]))
        X["3_0"] = b["x3_0"](self.pool(X["2_0"]))
        X["4_0"] = b["x4_0"](self.pool(X["3_0"]))

        for j in range(1, self.depth):
            for i in range(self.depth - j):
                prev_nodes = [X[f"{i}_{k}"] for k in range(j)]
                deeper_up = self.up(X[f"{i + 1}_{j - 1}"])
                cat = torch.cat(prev_nodes + [deeper_up], dim=1)
                X[f"{i}_{j}"] = b[f"x{i}_{j}"](cat)

        return self.outc(X[f"0_{self.depth - 1}"])
