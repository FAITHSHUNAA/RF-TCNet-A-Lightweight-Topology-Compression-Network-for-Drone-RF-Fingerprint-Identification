import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicFrequencyAttention(nn.Module):
    """动态频率注意力模块"""

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
        # 通道注意力
        ca = self.channel_attention(x)

        # 频率注意力
        batch, ch, freq, time = x.size()
        fa_input = x.mean(dim=-1)  # [B,C,F]
        fa = self.frequency_attention(fa_input).unsqueeze(-1)  # [B,C,F,1]

        return x * ca * fa


class EnergyTopologyPooling(nn.Module):
    """能量拓扑池化模块（关键修复）"""

    def __init__(self, pool_size=2, stride=2):  # 修改池化参数
        super().__init__()
        self.pool = nn.AvgPool2d(
            kernel_size=(1, pool_size),
            stride=(1, stride),
            padding=(0, pool_size // 2)  # 添加padding保持尺寸对齐
        )

    def forward(self, x):
        energy_map = torch.abs(x)
        weighted_x = x * energy_map
        return self.pool(weighted_x)


class RFTCNetBlock(nn.Module):
    """RF-TCNet基础模块（关键修复）"""

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        # 修改卷积参数保持尺寸对齐
        self.dw_conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3,
                      stride=1,  # 固定stride=1，降采样由池化完成
                      padding=1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True))

        self.dfa = DynamicFrequencyAttention(in_ch)

        self.pw_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch))

        # 统一降采样方式
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

        # 这里使用F.interpolate来确保residual和x的尺寸匹配
        if self.res_pool is not None:
            residual = self.res_pool(residual)

        # 调整residual的尺寸以匹配x
        residual = F.interpolate(residual, size=x.shape[2:], mode='bilinear', align_corners=False)

        return F.relu(x + residual)


class RF_TCNet(nn.Module):
    """完整的RF-TCNet模型"""

    def __init__(self, num_classes=25):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),  # 输入为单通道时频图
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )

        # 构建特征提取器
        self.features = nn.Sequential(
            RFTCNetBlock(16, 32, stride=2),
            RFTCNetBlock(32, 64),
            RFTCNetBlock(64, 128, stride=2),
            RFTCNetBlock(128, 256),
        )

        # 分类头
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, num_classes))

    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        return self.classifier(x)


# ----------------------
# 模型参数量验证
# ----------------------
if __name__ == "__main__":
    model = RF_TCNet(num_classes=25)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.3f}M")

    # 测试输入：batch_size=2, 1 channel, 128x128
    x = torch.randn(2, 1, 128, 128)
    try:
        out = model(x)
        print("Output shape:", out.shape)  # 预期输出: [2, 25]
    except Exception as e:
        print("Error:", e)
