import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicFrequencyAttention(nn.Module):
    """Dynamic Frequency Attention Module"""

    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1),
            nn.Sigmoid()
        )

        self.frequency_attention = nn.Sequential(
            nn.Conv1d(in_channels, in_channels // reduction_ratio, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels // reduction_ratio, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Channel Attention
        ca = self.channel_attention(x)

        # Frequency Attention
        batch, ch, freq, time = x.size()
        fa_input = x.mean(dim=-1)  # [B,C,F]
        fa = self.frequency_attention(fa_input).unsqueeze(-1)  # [B,C,F,1]

        return x * ca * fa


class EnergyTopologyPooling(nn.Module):
    """Energy topological pooling module"""

    def __init__(self, pool_size=2, stride=2):
        super().__init__()
        self.pool = nn.AvgPool2d(
            kernel_size=(1, pool_size),
            stride=(1, stride),
            padding=(0, pool_size // 2) 
        )

    def forward(self, x):
        energy_map = torch.abs(x)
        weighted_x = x * energy_map
        return self.pool(weighted_x)


class RFTCNetBlock(nn.Module):
    """RF-TCNet basic module"""

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        # Modify convolution parameters while maintaining size alignment
        self.dw_conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3,
                      stride=1, 
                      padding=1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True))

        self.dfa = DynamicFrequencyAttention(in_ch)

        self.pw_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch))

        # Uniform downsampling method
        if stride == 2:
            self.etp = EnergyTopologyPooling(pool_size=2, stride=2)
            self.res_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        else:
            self.etp = None
            self.res_pool = None

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.AvgPool2d(2, 2) if stride == 2 else nn.Identity()
            )

    def forward(self, x):
        residual = self.shortcut(x)

        x = self.dw_conv(x)
        x = self.dfa(x)
        x = self.pw_conv(x)

        if self.etp is not None:
            x = self.etp(x)

        # Use F.interpolate to ensure that the sizes of residual and x are matched.
        if self.res_pool is not None:
            residual = self.res_pool(residual)

        # Adjust the size of residual to match x
        residual = F.interpolate(residual, size=x.shape[2:], mode='bilinear', align_corners=False)

        return F.relu(x + residual)


class RF_TCNet(nn.Module):
    """The complete RF-TCNet model"""

    def __init__(self, num_classes=25):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), 
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )

        # Build a feature extractor
        self.features = nn.Sequential(
            RFTCNetBlock(16, 32, stride=2),
            RFTCNetBlock(32, 64),
            RFTCNetBlock(64, 128, stride=2),
            RFTCNetBlock(128, 256),
        )

        # Classification Head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, num_classes))

    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        return self.classifier(x)


# ----------------------
# Model parameter quantity verification
# ----------------------
if __name__ == "__main__":
    model = RF_TCNet(num_classes=25)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.3f}M")

    # test input：batch_size=2, 1 channel, 128x128
    x = torch.randn(2, 1, 128, 128)
    try:
        out = model(x)
        print("Output shape:", out.shape)
    except Exception as e:
        print("Error:", e)
