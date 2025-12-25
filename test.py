import torch
import torch.nn as nn


# 定义轻量化的基础卷积层
def Conv2d_BN(inp, oup, kernel_size, stride, padding, groups=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


# 定义 SE 注意力机制
class SqueezeExcite(nn.Module):
    def __init__(self, channels, reduction=0.25):
        super(SqueezeExcite, self).__init__()
        reduced_channels = max(1, int(channels * reduction))
        self.fc1 = nn.Conv2d(channels, reduced_channels, kernel_size=1)
        self.fc2 = nn.Conv2d(reduced_channels, channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        se = x.mean((2, 3), keepdim=True)  # 全局平均池化
        se = self.fc1(se)
        se = self.relu(se)
        se = self.fc2(se)
        return x * self.sigmoid(se)


# 定义 Residual 模块
class Residual(nn.Module):
    def __init__(self, module):
        super(Residual, self).__init__()
        self.module = module

    def forward(self, x):
        return x + self.module(x)


# 定义 InvertedResidual（倒残差块）
class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = stride == 1 and inp == oup
        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(Conv2d_BN(inp, hidden_dim, kernel_size=1, stride=1, padding=0))
        layers.extend([
            # dw
            Conv2d_BN(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


# 定义主 HybridBlock 模块
class HybridBlock(nn.Module):
    def __init__(self, inp, oup, hidden_dim, kernel_size, stride, use_se=True, use_hs=True):
        super(HybridBlock, self).__init__()
        self.identity = stride == 1 and inp == oup
        self.use_se = use_se

        # Token mixer (局部特征提取部分) - RepKKBlock部分
        if stride == 2:
            self.token_mixer = nn.Sequential(
                Conv2d_BN(inp, inp, kernel_size, stride, (kernel_size - 1) // 2, groups=inp),
                SqueezeExcite(inp, 0.25) if use_se else nn.Identity(),
                Conv2d_BN(inp, oup, kernel_size=1, stride=1, padding=0)
            )
        else:
            self.token_mixer = nn.Sequential(
                Conv2d_BN(inp, inp, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2, groups=inp),
                SqueezeExcite(inp, 0.25) if use_se else nn.Identity()
            )

        # 全局特征提取（使用 Block 的频域操作）
        self.global_mixer = Residual(nn.Sequential(
            Conv2d_BN(oup, hidden_dim, 1, 1, 0),
            nn.GELU() if use_hs else nn.ReLU(),
            Conv2d_BN(hidden_dim, oup, 1, 1, 0),
        ))

        # 倒残差结构，进一步提升特征提取能力
        self.channel_mixer = InvertedResidual(inp=oup, oup=oup, stride=1, expand_ratio=4)

    def forward(self, x):
        # Token mixer 部分（局部特征）
        x = self.token_mixer(x)

        # Global mixer 部分（全局特征）
        x = self.global_mixer(x)

        # Channel mixer 部分
        x = self.channel_mixer(x)

        return x


# 测试 HybridBlock
input_tensor = torch.randn(1, 64, 32, 32)  # 假设输入大小为 (batch, channels, height, width)
hybrid_block = HybridBlock(inp=64, oup=64, hidden_dim=128, kernel_size=3, stride=1, use_se=True, use_hs=True)
output = hybrid_block(input_tensor)
print(output.shape)  # 应输出 torch.Size([1, 64, 32, 32])
