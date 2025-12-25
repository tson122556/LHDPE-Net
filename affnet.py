import torch
import torch.nn as nn
import torch.nn.functional as F


class SELayer(nn.Module):
    """
    Squeeze-and-Excitation Layer: A lightweight attention mechanism for recalibrating channel-wise feature responses
    """
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.fc1 = nn.Conv2d(channel, channel // reduction, 1, bias=False)
        self.fc2 = nn.Conv2d(channel // reduction, channel, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = F.adaptive_avg_pool2d(x, (1, 1))  # Global Average Pooling
        y = self.fc1(y)
        y = F.relu(y, inplace=True)
        y = self.fc2(y)
        y = self.sigmoid(y)
        return x * y  # Channel-wise feature recalibration


class AFFNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, use_act=True, use_norm=True):
        super(AFFNetBlock, self).__init__()
        self.use_norm = use_norm
        self.use_act = use_act

        # 卷积层
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)

        # 批量归一化（可选）
        if self.use_norm:
            self.bn = nn.BatchNorm2d(out_channels)

        # ReLU 激活函数（可选）
        if self.use_act:
            self.act = nn.ReLU(inplace=True)

        # 注意力机制（使用Squeeze-and-Excitation）
        self.attention = SELayer(out_channels)  # Apply after the Conv layer

    def forward(self, x):
        # 应用傅里叶变换到输入
        x_freq = torch.fft.fft2(x)  # 进行二维傅里叶变换（频域）
        x_freq = torch.fft.fftshift(x_freq)  # 将零频率分量移到中心

        # 获取频域张量的幅度（实部）
        x_freq_real = torch.abs(x_freq)

        # 如果需要，可以在频域应用注意力机制
        x_freq_real = self.attention(x_freq_real)

        # 进行反傅里叶变换回到空间域
        x_spatial = torch.fft.ifft2(x_freq)  # 反傅里叶变换

        # 获取反傅里叶变换结果的幅度（实部）
        x_spatial = torch.abs(x_spatial)

        # 确保张量为float32类型，确保与卷积层兼容
        x_spatial = x_spatial.to(torch.float32)

        # 通过卷积层
        x_spatial = self.conv(x_spatial)

        # 批量归一化（如果使用的话）
        if self.use_norm:
            x_spatial = self.bn(x_spatial)

        # 激活函数（如果使用的话）
        if self.use_act:
            x_spatial = self.act(x_spatial)

        # 在空间域应用注意力机制
        x_spatial = self.attention(x_spatial)  # 确保注意力机制应用在卷积层之后

        return x_spatial


class AFFNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000):
        super(AFFNet, self).__init__()
        self.block1 = AFFNetBlock(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.block2 = AFFNetBlock(64, 128, kernel_size=3, stride=2, padding=1)
        self.block3 = AFFNetBlock(128, 256, kernel_size=3, stride=2, padding=1)
        self.block4 = AFFNetBlock(256, 512, kernel_size=3, stride=2, padding=1)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        # 全局平均池化
        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # 展平输出
        x = self.fc(x)
        return x


# 实例化并打印模型
if __name__ == "__main__":
    model = AFFNet(in_channels=3, num_classes=1000)
    print(model)

    # 测试模型
    input_tensor = torch.randn(1, 3, 224, 224)  # 随机输入张量（批量大小1，3个通道，224x224）
    output = model(input_tensor)
    print("输出形状:", output.shape)
