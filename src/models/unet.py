import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels: int = 11, num_classes: int = 6, base_channels: int = 64):
        super().__init__()
        self.base_channels = base_channels
        
        # Encoder (Downsampling)
        self.inc = ConvBlock(in_channels, base_channels)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(base_channels, base_channels * 2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(base_channels * 2, base_channels * 4))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(base_channels * 4, base_channels * 8))
        
        # Bottleneck
        self.bottleneck = nn.Sequential(nn.MaxPool2d(2), ConvBlock(base_channels * 8, base_channels * 16))
        
        # Decoder (Upsampling)
        self.up1 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, kernel_size=2, stride=2)
        self.conv_up1 = ConvBlock(base_channels * 16, base_channels * 8)
        
        self.up2 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.conv_up2 = ConvBlock(base_channels * 8, base_channels * 4)
        
        self.up3 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.conv_up3 = ConvBlock(base_channels * 4, base_channels * 2)
        
        self.up4 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.conv_up4 = ConvBlock(base_channels * 2, base_channels)
        
        # Output layer
        self.outc = nn.Conv2d(base_channels, num_classes, kernel_size=1)

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
