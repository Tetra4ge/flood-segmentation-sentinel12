import torch
import torch.nn as nn


class ResConvBlock(nn.Module):
    """Two 3x3 conv + BN + ReLU layers with a residual (identity/projection) shortcut and
    optional spatial dropout.

    This is an improved version of the original plain double-conv U-Net block: the residual
    shortcut eases gradient flow through the 4-level encoder/decoder (helps convergence in
    few epochs, which matters a lot when training on a very small scene count), and
    Dropout2d regularizes the otherwise heavily overparameterized 31M-parameter network
    against the tiny (14-scene) training set.
    """

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = nn.Identity()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        out = self.relu(self.conv(x) + self.shortcut(x))
        return self.drop(out)


class UNet(nn.Module):
    """U-Net baseline (Ronneberger et al., 2015), as evaluated in Table 1 of the IEW paper
    (F1 = 79.38%), extended with residual conv blocks and spatial dropout for better
    convergence/regularization on a small dataset (see ResConvBlock docstring).

    `base_channels=64` matches the paper-scale 31M-parameter configuration; use a smaller
    value (e.g. 32) for faster iteration on CPU.
    """

    def __init__(
        self,
        in_channels: int = 11,
        num_classes: int = 6,
        base_channels: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        c = base_channels

        # Encoder (Downsampling)
        self.inc = ResConvBlock(in_channels, c)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), ResConvBlock(c, c * 2, dropout))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), ResConvBlock(c * 2, c * 4, dropout))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), ResConvBlock(c * 4, c * 8, dropout))

        # Bottleneck
        self.bottleneck = nn.Sequential(nn.MaxPool2d(2), ResConvBlock(c * 8, c * 16, dropout))

        # Decoder (Upsampling)
        self.up1 = nn.ConvTranspose2d(c * 16, c * 8, kernel_size=2, stride=2)
        self.conv_up1 = ResConvBlock(c * 16, c * 8, dropout)

        self.up2 = nn.ConvTranspose2d(c * 8, c * 4, kernel_size=2, stride=2)
        self.conv_up2 = ResConvBlock(c * 8, c * 4, dropout)

        self.up3 = nn.ConvTranspose2d(c * 4, c * 2, kernel_size=2, stride=2)
        self.conv_up3 = ResConvBlock(c * 4, c * 2, dropout)

        self.up4 = nn.ConvTranspose2d(c * 2, c, kernel_size=2, stride=2)
        self.conv_up4 = ResConvBlock(c * 2, c, dropout)

        # Output layer
        self.outc = nn.Conv2d(c, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        # Bottleneck
        b = self.bottleneck(x4)

        # Decoder
        u1 = self.up1(b)
        u1 = torch.cat([u1, x4], dim=1)
        u1 = self.conv_up1(u1)

        u2 = self.up2(u1)
        u2 = torch.cat([u2, x3], dim=1)
        u2 = self.conv_up2(u2)

        u3 = self.up3(u2)
        u3 = torch.cat([u3, x2], dim=1)
        u3 = self.conv_up3(u3)

        u4 = self.up4(u3)
        u4 = torch.cat([u4, x1], dim=1)
        u4 = self.conv_up4(u4)

        logits = self.outc(u4)
        return logits
