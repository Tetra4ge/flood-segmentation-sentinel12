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
    def __init__(self, in_channels: int = 11, num_classes: int = 6):
        super().__init__()
        
        # Encoder (Downsampling)
        self.inc = ConvBlock(in_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(256, 512))
        
        # Bottleneck
        self.bottleneck = nn.Sequential(nn.MaxPool2d(2), ConvBlock(512, 1024))
        
        # Decoder (Upsampling)
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv_up1 = ConvBlock(1024, 512)
        
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv_up2 = ConvBlock(512, 256)
        
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up3 = ConvBlock(256, 128)
        
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up4 = ConvBlock(128, 64)
        
        # Output layer
        self.outc = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)        # (B, 64, H, W)
        x2 = self.down1(x1)     # (B, 128, H/2, W/2)
        x3 = self.down2(x2)     # (B, 256, H/4, W/4)
        x4 = self.down3(x3)     # (B, 512, H/8, W/8)
        
        # Bottleneck
        b = self.bottleneck(x4) # (B, 1024, H/16, W/16)
        
        # Decoder
        u1 = self.up1(b)        # (B, 512, H/8, W/8)
        # Concatenate skip connections
        u1 = torch.cat([u1, x4], dim=1) # (B, 1024, H/8, W/8)
        u1 = self.conv_up1(u1)  # (B, 512, H/8, W/8)
        
        u2 = self.up2(u1)        # (B, 256, H/4, W/4)
        u2 = torch.cat([u2, x3], dim=1) # (B, 512, H/4, W/4)
        u2 = self.conv_up2(u2)  # (B, 256, H/4, W/4)
        
        u3 = self.up3(u2)        # (B, 128, H/2, W/2)
        u3 = torch.cat([u3, x2], dim=1) # (B, 256, H/2, W/2)
        u3 = self.conv_up3(u3)  # (B, 128, H/2, W/2)
        
        u4 = self.up4(u3)        # (B, 64, H, W)
        u4 = torch.cat([u4, x1], dim=1) # (B, 128, H, W)
        u4 = self.conv_up4(u4)  # (B, 64, H, W)
        
        logits = self.outc(u4)  # (B, num_classes, H, W)
        return logits
