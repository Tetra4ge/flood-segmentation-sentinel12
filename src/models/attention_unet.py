import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Two 3x3 conv + BN + ReLU layers, as in the original U-Net encoder/decoder blocks."""

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


class AttentionGate(nn.Module):
    """Additive attention gate (Oktay et al., 2018, 'Attention U-Net: Learning Where to Look
    for the Pancreas'). Suppresses skip-connection activations in regions irrelevant to the
    coarser decoder signal `g`, before concatenation.

    F_g: channels of the gating signal (from the decoder, coarser resolution, upsampled to match x)
    F_l: channels of the skip connection `x` (from the encoder)
    F_int: channels of the intermediate bottleneck (typically F_l // 2)
    """

    def __init__(self, f_g: int, f_l: int, f_int: int):
        super().__init__()
        self.w_g = nn.Sequential(
            nn.Conv2d(f_g, f_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(f_int),
        )
        self.w_x = nn.Sequential(
            nn.Conv2d(f_l, f_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(f_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(f_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.w_g(g)
        x1 = self.w_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class UpAttnBlock(nn.Module):
    """Upsample the decoder feature map, gate the skip connection through an AttentionGate,
    concatenate, and fuse with a ConvBlock."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.attn = AttentionGate(f_g=out_channels, f_l=skip_channels, f_int=skip_channels // 2)
        self.conv = ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        skip = self.attn(g=x, x=skip)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class AttentionUNet(nn.Module):
    """Attention U-Net: standard U-Net encoder/bottleneck with attention-gated skip connections
    in the decoder (Oktay et al., 2018), as evaluated in Table 1 of the IEW paper (F1 = 77.77%)."""

    def __init__(self, in_channels: int = 11, num_classes: int = 6, base_channels: int = 32):
        super().__init__()
        c = base_channels

        # Encoder
        self.inc = ConvBlock(in_channels, c)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(c, c * 2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(c * 2, c * 4))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(c * 4, c * 8))

        # Bottleneck
        self.bottleneck = nn.Sequential(nn.MaxPool2d(2), ConvBlock(c * 8, c * 16))

        # Decoder with attention-gated skips
        self.up1 = UpAttnBlock(c * 16, c * 8, c * 8)
        self.up2 = UpAttnBlock(c * 8, c * 4, c * 4)
        self.up3 = UpAttnBlock(c * 4, c * 2, c * 2)
        self.up4 = UpAttnBlock(c * 2, c, c)

        self.outc = nn.Conv2d(c, num_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        b = self.bottleneck(x4)

        u1 = self.up1(b, x4)
        u2 = self.up2(u1, x3)
        u3 = self.up3(u2, x2)
        u4 = self.up4(u3, x1)

        return self.outc(u4)
