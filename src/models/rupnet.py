import torch
import torch.nn as nn

class Conv2D(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1, stride=1, dilation=1, bias=False, act=True):
        super().__init__()
        self.act = act
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias),
            nn.BatchNorm2d(out_c)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.act:
            x = self.relu(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c)
        )
        self.shortcut = nn.Sequential()
        if in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_c)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x) + self.shortcut(x))

class DecoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.res = ResidualBlock(in_c + out_c, out_c)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.res(x)
        return x

class UpBlock(nn.Module):
    def __init__(self, in_c, out_c, scale):
        super().__init__()
        self.up = nn.Upsample(scale_factor=scale, mode="bilinear", align_corners=True)
        self.res = ResidualBlock(in_c, out_c)

    def forward(self, x):
        x = self.up(x)
        x = self.res(x)
        return x

class RUPNet(nn.Module):
    def __init__(self, in_channels: int = 11, num_classes: int = 6):
        super().__init__()
        
        # 1. Custom CNN Encoder
        self.init_conv = Conv2D(in_channels, 32, kernel_size=3, padding=1)
        
        self.enc1_1 = ResidualBlock(32, 64)
        self.down1 = nn.MaxPool2d(2)
        self.enc1_2 = ResidualBlock(64, 64)
        self.down2 = nn.MaxPool2d(2) # scale: H/4
        
        self.enc2 = ResidualBlock(64, 128)
        self.down3 = nn.MaxPool2d(2) # scale: H/8
        
        self.enc3 = ResidualBlock(128, 256)
        self.down4 = nn.MaxPool2d(2) # scale: H/16
        
        # 2. Channel Reduction (to 64 channels)
        self.c1 = Conv2D(64, 64, kernel_size=1, padding=0)
        self.c2 = Conv2D(128, 64, kernel_size=1, padding=0)
        self.c3 = Conv2D(256, 64, kernel_size=1, padding=0)
        
        # 3. Decoders
        self.d1 = DecoderBlock(64, 64)
        self.d2 = DecoderBlock(64, 64)
        
        # Decoder 3 / Up-sampling blocks
        self.d3 = UpBlock(64, 64, scale=4)
        self.u1 = UpBlock(64, 64, scale=4)
        self.u2 = UpBlock(64, 64, scale=8)
        self.u3 = UpBlock(64, 64, scale=16)
        
        # 4. Fusion and Head
        self.fusion = ResidualBlock(256, 64)
        self.outc = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, inputs):
        # Encoder forward pass
        x = self.init_conv(inputs)
        
        e1 = self.enc1_1(x)
        e1 = self.down1(e1)
        e1 = self.enc1_2(e1)
        e1 = self.down2(e1) # shape: (B, 64, H/4, W/4)
        
        e2 = self.enc2(e1)
        e2 = self.down3(e2) # shape: (B, 128, H/8, W/8)
        
        e3 = self.enc3(e2)
        e3 = self.down4(e3) # shape: (B, 256, H/16, W/16)
        
        # Channel reduction to 64 channels
        c1 = self.c1(e1)
        c2 = self.c2(e2)
        c3 = self.c3(e3)
        
        # Decoders
        d1 = self.d1(c3, c2) # input from c3 (H/16), skip from c2 (H/8) -> output shape: (B, 64, H/8, W/8)
        d2 = self.d2(d1, c1) # input from d1 (H/8), skip from c1 (H/4) -> output shape: (B, 64, H/4, W/4)
        
        # Upsampling to original size H, W
        d3 = self.d3(d2) # shape: (B, 64, H, W)
        u1 = self.u1(c1) # shape: (B, 64, H, W)
        u2 = self.u2(c2) # shape: (B, 64, H, W)
        u3 = self.u3(c3) # shape: (B, 64, H, W)
        
        # Concat outputs of all up-blocks
        fused = torch.cat([d3, u1, u2, u3], dim=1) # shape: (B, 256, H, W)
        fused = self.fusion(fused) # shape: (B, 64, H, W)
        
        logits = self.outc(fused) # shape: (B, num_classes, H, W)
        return logits
