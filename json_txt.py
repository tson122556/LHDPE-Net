import torch
import torch.nn as nn


class LSKmodule(nn.Module):
    def __init__(self, dim, input_dim=None):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.convl = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv0_s = nn.Conv2d(dim, dim//2, 1)
        self.conv1_s = nn.Conv2d(dim, dim//2, 1)
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv_m = nn.Conv2d(dim//2, dim, 1)

    def forward(self, x):
        attn1 = self.conv0(x)
        attn2 = self.convl(attn1)

        attn1 = self.conv0_s(attn1)
        attn2 = self.conv1_s(attn2)

        attn = torch.cat([attn1, attn2], dim=1)
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)
        sig = self.conv_squeeze(agg).sigmoid()
        attn = attn1 * sig[:,0,:,:].unsqueeze(1) + attn2 * sig[:,1,:,:].unsqueeze(1)
        attn = self.conv_m (attn)
        return x * attn

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        print("Input shape:", x.shape)
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        # 打印输出形状
        print("Output shape after forward:", x.shape)
        return x



    def forward_fuse(self, x):
        # 打印输入x的形状
        print("Input shape for forward_fuse:", x.shape)
        x = self.act(self.conv(x))
        # 打印输出形状
        print("Output shape after forward_fuse:", x.shape)
        return x

dim = 3
lsk_module = LSKmodule(dim)
#lsk_module = Conv(c2)

input_tensor = torch.randn(1, dim, 64, 64)  # 假设batch大小为1，其他维度根据需要设置

# 前向传播并打印输出形状
output = lsk_module(input_tensor)
print("Output shape:", output.shape)


c1, c2 = 3, 64
conv_mod = Conv(c1, c2)

# 创建一个随机初始化的输入张量，假设batch大小为1，H和W分别为64
input_tensor = torch.randn(1, c1, 64, 64)

# 使用forward方法
output = conv_mod(input_tensor)
# 使用forward_fuse方法
output_fuse = conv_mod.forward_fuse(input_tensor)