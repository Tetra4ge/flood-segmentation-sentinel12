import json

# Load RUPNet notebook
with open('03_rupnet_backbone.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Cell 0: Title
nb['cells'][0]['source'] = [
    "# Inland Excess Water (IEW) Monitoring — TransUNet\n",
    "\n",
    "Consolidated, self-contained notebook implementing **TransUNet**, a transformer-based backbone for semantic segmentation.\n"
]

# Cell 13: Architecture Title
nb['cells'][13]['source'] = [
    "## 5. Model Architecture — TransUNet\n",
    "Implementation of TransUNet, combining a CNN feature extractor, a Vision Transformer bottleneck, and a U-Net style decoder.\n",
    "Note: Ensure that patch_size in configs/data.yaml is large enough (e.g., 128) for the Transformer to have meaningful spatial context."
]

# Cell 14: Model Architecture
transunet_code = """import torch
import torch.nn as nn
import torch.nn.functional as F

# A simple fallback rearrange if einops is not installed
def rearrange_tensor(x):
    B, C, H, W = x.shape
    return x.view(B, C, H * W).transpose(1, 2)

def rearrange_back(x, H, W):
    B, L, C = x.shape
    return x.transpose(1, 2).view(B, C, H, W)

class CNNEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Simple ResNet-style downsampling for feature extraction
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(2)
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(2)
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.pool3 = nn.MaxPool2d(2)
        
    def forward(self, x):
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        return p3, [c1, c2, c3]

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim)
        )
        
    def forward(self, x):
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x

class VisionTransformer(nn.Module):
    def __init__(self, in_channels, embed_dim, num_heads, num_layers, mlp_dim, grid_size):
        super().__init__()
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=1)
        
        # Absolute position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, grid_size[0] * grid_size[1], embed_dim))
        
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_dim) for _ in range(num_layers)
        ])
        
    def forward(self, x):
        # x is (B, C, H, W)
        x = self.patch_embed(x)
        B, C, H, W = x.shape
        x = rearrange_tensor(x)
        
        x = x + self.pos_embed
        for block in self.blocks:
            x = block(x)
            
        x = rearrange_back(x, H, W)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels // 2 + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class TransUNet(nn.Module):
    def __init__(self, in_channels=11, num_classes=6, img_size=128):
        super().__init__()
        
        # CNN Encoder
        cnn_out_channels = 256
        self.encoder = CNNEncoder(in_channels, cnn_out_channels)
        
        # Transformer Bottleneck
        # 3 MaxPools in CNN => reduction factor of 8
        grid_size = img_size // 8
        embed_dim = 512
        self.vit = VisionTransformer(
            in_channels=cnn_out_channels,
            embed_dim=embed_dim,
            num_heads=8,
            num_layers=4,
            mlp_dim=1024,
            grid_size=(grid_size, grid_size)
        )
        
        # Decoder
        self.dec3 = DecoderBlock(embed_dim, 256, 128)
        self.dec2 = DecoderBlock(128, 128, 64)
        self.dec1 = DecoderBlock(64, 64, 32)
        
        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)
        
    def forward(self, x):
        # x is (B, 11, H, W)
        
        # CNN encode
        x, skips = self.encoder(x) 
        
        # Transformer bottleneck
        x = self.vit(x)
        
        # Decoder
        x = self.dec3(x, skips[2]) 
        x = self.dec2(x, skips[1])
        x = self.dec1(x, skips[0]) 
        
        x = self.final_conv(x)
        return x
"""
nb['cells'][14]['source'] = [line + '\n' for line in transunet_code.split('\n')]

# Cell 17: Model Instantiation
nb['cells'][17]['source'] = [
    "device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n",
    "\n",
    "model = TransUNet(in_channels=11, num_classes=6, img_size=128).to(device)\n",
    "\n",
    "optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)\n",
    "\n",
    "criterion_gdl = GeneralizedDiceLoss(ignore_classes=[10, 11])\n",
    "criterion_wfl = WeightedFocalLoss(alpha=class_weights.to(device), ignore_classes=[10, 11])\n",
    "\n",
    "def combined_loss(pred, target):\n",
    "    return criterion_gdl(pred, target) + criterion_wfl(pred, target)\n"
]

# Cell 24: Model Name
nb['cells'][24]['source'] = [
    "MODEL_NAME = \"TransUNet\"\n",
    "\n",
    "plt.figure(figsize=(10, 5))\n",
    "plt.plot(train_losses, label='Train Loss')\n",
    "plt.plot(val_losses, label='Validation Loss')\n",
    "plt.title(f\"{MODEL_NAME} Loss Curve\")\n",
    "plt.xlabel(\"Epoch\")\n",
    "plt.ylabel(\"Loss\")\n",
    "plt.legend()\n",
    "plt.grid(True)\n",
    "plt.show()\n"
]

with open('08_transunet_backbone.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
