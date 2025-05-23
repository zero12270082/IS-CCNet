import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.arch_util import to_2tuple, trunc_normal_
from torch.utils import checkpoint


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """在残差块的主路径上随机丢弃路径（随机深度）。

    参数：
        x: 输入张量
        drop_prob: 丢弃概率
        training: 是否处于训练模式
    """
    if drop_prob == 0. or not training:  # 如果丢弃概率为 0 或者不在训练模式下
        return x  # 直接返回输入 x
    keep_prob = 1 - drop_prob  # 计算保留概率
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # 定义形状以支持不同维度的张量
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)  # 生成随机张量
    random_tensor.floor_()  # 对随机张量进行二值化
    output = x.div(keep_prob) * random_tensor  # 按保留概率返回输出
    return output  # 返回输出张量


class HIN(nn.Module):
    """半实例归一化模块。

    参数：
        in_size (int): 输入特征通道数。
        out_size (int): 输出特征通道数。
    """

    def __init__(self, in_size, out_size):  # 初始化函数
        super(HIN, self).__init__()  # 初始化 nn.Module
        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)  # 第一个卷积层
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)  # 第二个卷积层
        self.norm = nn.InstanceNorm2d(out_size // 2, affine=True)  # 实例归一化层
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)  # 残差连接的卷积层

    def forward(self, x):  # 前向传播函数
        out = self.conv_1(x)  # 通过第一个卷积层
        out_1, out_2 = torch.chunk(out, 2, dim=1)  # 切分输出为两部分
        out = torch.cat([self.norm(out_1), out_2], dim=1)  # 对第一部分进行归一化并与第二部分拼接
        out = F.relu(out)  # 应用 ReLU 激活函数
        out = F.relu(self.conv_2(out))  # 通过第二个卷积层后激活
        out = self.identity(x) + out  # 加上残差连接
        return out  # 返回输出


class ChannelAttention(nn.Module):
    """通道注意力机制。

    参数：
        num_feat (int): 中间特征的通道数。
        squeeze_factor (int): 通道压缩因子。默认值：16。
    """

    def __init__(self, num_feat, squeeze_factor=16):  # 初始化函数
        super(ChannelAttention, self).__init__()  # 初始化 nn.Module
        # scam模块
        self.attention = nn.Sequential(  # 定义顺序容器
            nn.AdaptiveAvgPool2d(1),  # 自适应平均池化层
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),  # 卷积层降低通道数
            nn.ReLU(inplace=True),  # ReLU 激活函数
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),  # 卷积层恢复通道数
            nn.Sigmoid())  # Sigmoid 激活函数

    def forward(self, x):  # 前向传播函数
        y = self.attention(x)  # 计算注意力
        return x * y  # 将原始输入与注意力相乘以增强特征


class CAB(nn.Module):
    """通道增强块。"""

    def __init__(self, num_feat, compress_ratio=3, squeeze_factor=30):  # 初始化函数
        super(CAB, self).__init__()  # 初始化 nn.Module
        self.cab = nn.Sequential(  # 定义顺序容器
            nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),  # 卷积层进行压缩
            nn.GELU(),  # GELU 激活函数
            nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),  # 卷积层恢复通道数
            ChannelAttention(num_feat, squeeze_factor)  # 通过通道注意力模块
        )

    def forward(self, x):  # 前向传播函数
        return self.cab(x)  # 返回增强后的特征


class DropPath(nn.Module):
    """在残差块的主路径上随机丢弃路径（随机深度）。
    来源: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py
    """

    def __init__(self, drop_prob=None):  # 初始化函数
        super(DropPath, self).__init__()  # 初始化 nn.Module
        self.drop_prob = drop_prob  # 保存丢弃概率

    def forward(self, x):  # 前向传播函数
        return drop_path(x, self.drop_prob, self.training)  # 调用 drop_path 函数


class Mlp(nn.Module):
    """多层感知机模块。"""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):  # 初始化函数
        super().__init__()  # 初始化 nn.Module
        out_features = out_features or in_features  # 输出特征数可选择
        hidden_features = hidden_features or in_features  # 隐藏特征数可选择
        self.fc1 = nn.Linear(in_features, hidden_features)  # 第一个全连接层
        self.act = act_layer()  # 激活函数
        self.fc2 = nn.Linear(hidden_features, out_features)  # 第二个全连接层
        self.drop = nn.Dropout(drop)  # Dropout 层以防止过拟合

    def forward(self, x):  # 前向传播函数
        x = self.fc1(x)  # 通过第一个全连接层
        x = self.act(x)  # 激活
        x = self.drop(x)  # 应用 Dropout
        x = self.fc2(x)  # 通过第二个全连接层
        x = self.drop(x)  # 应用 Dropout
        return x  # 返回输出


def window_partition(x, window_size):
    """将输入张量分区为窗口。

    参数：
        x: 输入张量 (b, h, w, c)
        window_size (int): 窗口大小

    返回：
        windows: 窗口张量 (num_windows*b, window_size, window_size, c)
    """
    b, h, w, c = x.shape  # 获取输入张量尺寸

    # 检查 h 和 w 是否可以被 window_size 整除
    if h % window_size != 0 or w % window_size != 0:
        raise ValueError(f"Height {h} and width {w} must be divisible by window size {window_size}.")

    # 进行张量重塑
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)  # 将张量重塑
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, c)  # 重新排列并展开

    return windows  # 返回窗口张量



def window_reverse(windows, window_size, h, w):
    """将窗口张量恢复为原始形状。

    参数：
        windows: 窗口张量 (num_windows*b, window_size, window_size, c)
        window_size (int): 窗口大小
        h (int): 图像高度
        w (int): 图像宽度

    返回：
        x: 恢复的张量 (b, h, w, c)
    """
    b = int(windows.shape[0] / (h * w / window_size / window_size))  # 计算批次大小
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)  # 重塑张量
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)  # 再次排列
    return x  # 返回恢复后的张量


class WindowAttention(nn.Module):
    """基于窗口的多头自注意力模块 (W-MSA) ，带有相对位置偏差。
    支持偏移窗口和非偏移窗口。

    参数：
        dim (int): 输入通道数量。
        window_size (tuple[int]): 窗口的高度和宽度。
        num_heads (int): 注意力头的数量。
        qkv_bias (bool, optional): 是否将可学习的偏置添加到查询、键、值中。默认值: True
        qk_scale (float | None, optional): 如果设置，则覆盖默认的 qk 缩放。
        attn_drop (float, optional): 注意力权重的丢弃比率。默认值: 0.0
        proj_drop (float, optional): 输出的丢弃比率。默认值: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):  # 初始化函数
        super().__init__()  # 初始化 nn.Module
        self.dim = dim  # 输入通道数
        self.window_size = window_size  # 窗口大小
        self.num_heads = num_heads  # 注意力头数
        head_dim = dim // num_heads  # 每个头的维度
        self.scale = qk_scale or head_dim ** -0.5  # 计算缩放因子

        # 定义相对位置偏差参数表
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 位置偏差表

        # 获取每个窗口内部的成对相对位置索引
        coords_h = torch.arange(self.window_size[0])  # 高度坐标
        coords_w = torch.arange(self.window_size[1])  # 宽度坐标
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 生成网格
        coords_flatten = torch.flatten(coords, 1)  # 扁平化坐标
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 计算相对坐标
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # 重新排列
        relative_coords[:, :, 0] += self.window_size[0] - 1  # 偏移量调整
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1  # 索引计算
        relative_position_index = relative_coords.sum(-1)  # 获得位置索引
        self.register_buffer('relative_position_index', relative_position_index)  # 注册相对位置索引

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)  # 查询、键、值的线性映射
        self.attn_drop = nn.Dropout(attn_drop)  # 注意力丢弃层
        self.proj = nn.Linear(dim, dim)  # 输出映射层
        self.proj_drop = nn.Dropout(proj_drop)  # 输出丢弃层
        trunc_normal_(self.relative_position_bias_table, std=.02)  # 初始化相对位置偏差
        self.softmax = nn.Softmax(dim=-1)  # Softmax 层

    def forward(self, x, mask=None):  # 前向传播函数
        """
        参数：
            x: 输入特征，形状为 (num_windows*b, n, c)
            mask: (0/-inf) 掩码，形状为 (num_windows, Wh*Ww, Wh*Ww) 或 None
        """
        b_, n, c = x.shape  # 获取输入尺寸
        qkv = self.qkv(x).reshape(b_, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)  # 计算 qkv
        q, k, v = qkv[0], qkv[1], qkv[2]  # 分别提取 q、k、v

        q = q * self.scale  # 应用缩放
        attn = (q @ k.transpose(-2, -1))  # 计算注意力权重

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # 获得相对位置偏差
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # 重新排列
        attn = attn + relative_position_bias.unsqueeze(0)  # 加入相对位置偏差

        if mask is not None:  # 如果有掩码
            nw = mask.shape[0]  # 获取窗口数
            attn = attn.view(b_ // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)  # 应用掩码
            attn = attn.view(-1, self.num_heads, n, n)  # 重新排列
            attn = self.softmax(attn)  # 归一化
        else:
            attn = self.softmax(attn)  # 归一化

        attn = self.attn_drop(attn)  # 应用注意力丢弃

        x = (attn @ v).transpose(1, 2).reshape(b_, n, c)  # 计算输出
        x = self.proj(x)  # 映射到输出维度
        x = self.proj_drop(x)  # 应用输出丢弃
        return x  # 返回输出

    def extra_repr(self) -> str:  # 返回类的额外信息
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, n):  # 计算 FLOPS
        # 计算窗口中 token 长度为 n 的 flop 数
        flops = 0
        # qkv = self.qkv(x)
        flops += n * self.dim * 3 * self.dim  # qkv 计算量
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * n * (self.dim // self.num_heads) * n  # 注意力计算量
        #  x = (attn @ v)
        flops += self.num_heads * n * n * (self.dim // self.num_heads)  # 合并计算量
        # x = self.proj(x)
        flops += n * self.dim * self.dim  # 输出映射
        return flops  # 返回总计算量


class SwinTransformerBlock(nn.Module):
    """Swin Transformer 块。

    参数：
        dim (int): 输入通道数。
        input_resolution (tuple[int]): 输入分辨率。
        num_heads (int): 注意力头数量。
        window_size (int): 窗口大小。
        shift_size (int): SW-MSA 的偏移大小。
        mlp_ratio (float): MLP 隐藏层维度与嵌入层维度的比率。
        qkv_bias (bool, optional): 如果为 True，则在查询、键、值中添加可学习的偏置。默认值: True
        qk_scale (float | None, optional): 如果设置，则覆盖默认的 qk 缩放。
        drop (float, optional): 丢弃率。默认值: 0.0
        attn_drop (float, optional): 注意力丢弃率。默认值: 0.0
        drop_path (float, optional): 随机深度率。默认值: 0.0
        act_layer (nn.Module, optional): 激活层。默认值: nn.GELU
        norm_layer (nn.Module, optional): 归一化层。默认值: nn.LayerNorm
    """

    def calculate_mask(self, x_size):
        """计算 SW-MSA 的注意力掩码。

        参数：
            x_size: 输入特征图的大小 (h, w)

        返回：
            attn_mask: 计算得到的注意力掩码
        """
        h, w = x_size  # 获取输入特征图的高度和宽度
        img_mask = torch.zeros((1, h, w, 1))  # 初始化图像掩码为零
        h_slices = (
        slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))  # 高度切片
        w_slices = (
        slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))  # 宽度切片
        cnt = 0  # 初始化计数器
        for h in h_slices:  # 遍历所有高度切片
            for w in w_slices:  # 遍历所有宽度切片
                img_mask[:, h, w, :] = cnt  # 在掩码中写入计数值
                cnt += 1  # 更新计数器

        mask_windows = window_partition(img_mask, self.window_size)  # 将掩码分区为窗口
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)  # 重新塑形为合适的形状
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # 计算注意力掩码
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0,
                                                                                     float(0.0))  # 应用掩码填充值

        return attn_mask  # 返回计算得到的注意力掩码

    def __init__(self,
                 dim,
                 input_resolution,
                 num_heads,
                 window_size=7,
                 shift_size=0,
                 compress_ratio=3,
                 squeeze_factor=30,
                 cab_scale=0.01,
                 hin_scale=0.01,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):  # 初始化函数
        super().__init__()  # 初始化 nn.Module
        self.dim = dim  # 输入通道数
        self.input_resolution = input_resolution  # 输入分辨率
        self.num_heads = num_heads  # 注意力头数量
        self.window_size = window_size  # 窗口大小
        self.shift_size = shift_size  # 偏移大小
        self.mlp_ratio = mlp_ratio  # MLP 比率
        if min(self.input_resolution) <= self.window_size:  # 如果窗口大小大于输入分辨率
            # 不进行窗口划分
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, 'shift_size must in 0-window_size'  # 确保偏移大小合法

        self.norm1 = norm_layer(dim)  # 第一个归一化层
        self.attn = WindowAttention(  # 定义窗口注意力模块
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop)

        self.cab_scale = cab_scale  # CAB 的缩放因子
        self.conv_block = CAB(num_feat=dim, compress_ratio=compress_ratio, squeeze_factor=squeeze_factor)  # CAB 模块
        self.hin_scale = hin_scale  # HIN 的缩放因子
        self.hin = HIN(in_size=dim, out_size=dim)  # HIN 模块
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()  # 随机丢弃路径
        self.norm2 = norm_layer(dim)  # 第二个归一化层
        mlp_hidden_dim = int(dim * mlp_ratio)  # MLP 隐藏层维度
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)  # MLP 模块

        if self.shift_size > 0:  # 如果偏移大小大于 0
            attn_mask = self.calculate_mask(self.input_resolution)  # 计算注意力掩码
        else:
            attn_mask = None  # 无掩码

        self.register_buffer('attn_mask', attn_mask)  # 注册注意力掩码

    def forward(self, x, x_size):
        """前向传播函数。

        参数：
            x: 输入特征张量
            x_size: 输入特征的大小 (h, w)

        返回：
            x: 经过多个操作处理后的输出
        """
        h, w = x_size  # 获取输入数据的高度和宽度
        b, _, c = x.shape  # 获取输入的批次大小和通道数

        shortcut = x  # 保存输入特征以实现残差连接
        x = self.norm1(x)  # 对输入进行第一次归一化
        x = x.view(b, h, w, c)  # 重新塑形为 (batch_size, height, width, channels)

        x_permute = x.permute(0, 3, 1, 2)  # 调整维度顺序为 (batch_size, channels, height, width)

        # CAB 模块
        conv_x = self.conv_block(x_permute)  # 通过 CAB 模块处理
        conv_x = conv_x.permute(0, 2, 3, 1).contiguous().view(b, h * w, c)  # 重新调整维度返回到 (batch_size, seq_len, channels)

        # HIN 模块
        hin = self.hin(x_permute)  # 通过 HIN 模块
        hin = hin.permute(0, 2, 3, 1).contiguous().view(b, h * w, c)  # 重新调整维度返回

        # 进行周期性偏移
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))  # 进行周期性偏移
        else:
            shifted_x = x  # 若无偏移则直接使用

        # 将输入划分为窗口
        x_windows = window_partition(shifted_x, self.window_size)  # 将数据划分为窗口
        x_windows = x_windows.view(-1, self.window_size * self.window_size, c)  # 重新塑形为合适的形状

        # W-MSA/SW-MSA（支持测试时图像尺寸是窗口大小的倍数）
        if self.input_resolution == x_size:
            attn_windows = self.attn(x_windows, mask=self.attn_mask)  # 计算注意力窗口
        else:
            attn_windows = self.attn(x_windows, mask=self.calculate_mask(x_size).to(x.device))  # 计算注意力窗口并应用掩码

        # 合并窗口
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, c)  # 将注意力窗口调整为合适的形状
        shifted_x = window_reverse(attn_windows, self.window_size, h, w)  # 逆向恢复窗口

        # 逆向周期性偏移
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))  # 逆向偏移
        else:
            x = shifted_x  # 若无偏移则直接使用
        x = x.view(b, h * w, c)  # 重新塑形返回到 (batch_size, height * width, channels)

        # 前馈网络
        x = shortcut + self.drop_path(x) + conv_x * self.cab_scale  # 加入卷积输出和残差连接
        x = x + self.drop_path(self.mlp(self.norm2(x)))  # 通过 MLP 前馈网络
        x = x + self.hin_scale * hin  # 加入 HIN 输出

        return x  # 返回最终输出

def calculate_mask(self, x_size):
    """计算 SW-MSA 的注意力掩码。

    参数：
        x_size: 输入特征图的大小 (h, w)

    返回：
        attn_mask: 计算得到的注意力掩码
    """
    h, w = x_size  # 获取输入特征图的高度和宽度
    img_mask = torch.zeros((1, h, w, 1))  # 初始化图像掩码为零
    h_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))  # 高度切片
    w_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))  # 宽度切片
    cnt = 0  # 初始化计数器
    for h in h_slices:  # 遍历所有高度切片
        for w in w_slices:  # 遍历所有宽度切片
            img_mask[:, h, w, :] = cnt  # 在掩码中写入计数值
            cnt += 1  # 更新计数器

    mask_windows = window_partition(img_mask, self.window_size)  # 将掩码分区为窗口
    mask_windows = mask_windows.view(-1, self.window_size * self.window_size)  # 重新塑形为合适的形状
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # 计算注意力掩码
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))  # 应用掩码填充值

    return attn_mask  # 返回计算得到的注意力掩码


def forward(self, x, x_size):
    """前向传播函数。

    参数：
        x: 输入特征张量
        x_size: 输入特征的大小 (h, w)

    返回：
        x: 经过多个操作处理后的输出
    """
    h, w = x_size  # 获取输入数据的高度和宽度
    b, _, c = x.shape  # 获取输入的批次大小和通道数

    shortcut = x  # 保存输入特征以实现残差连接
    x = self.norm1(x)  # 对输入进行第一次归一化
    x = x.view(b, h, w, c)  # 重新塑形为 (batch_size, height, width, channels)

    x_permute = x.permute(0, 3, 1, 2)  # 调整维度顺序为 (batch_size, channels, height, width)

    # CAB 模块
    conv_x = self.conv_block(x_permute)  # 通过 CAB 模块处理
    conv_x = conv_x.permute(0, 2, 3, 1).contiguous().view(b, h * w, c)  # 重新调整维度返回到 (batch_size, seq_len, channels)

    # HIN 模块
    hin = self.hin(x_permute)  # 通过 HIN 模块
    hin = hin.permute(0, 2, 3, 1).contiguous().view(b, h * w, c)  # 重新调整维度返回

    # 进行周期性偏移
    if self.shift_size > 0:
        shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))  # 进行周期性偏移
    else:
        shifted_x = x  # 若无偏移则直接使用

    # 将输入划分为窗口
    x_windows = window_partition(shifted_x, self.window_size)  # 将数据划分为窗口
    x_windows = x_windows.view(-1, self.window_size * self.window_size, c)  # 重新塑形为合适的形状

    # W-MSA/SW-MSA（支持测试时图像尺寸是窗口大小的倍数）
    if self.input_resolution == x_size:
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # 计算注意力窗口
    else:
        attn_windows = self.attn(x_windows, mask=self.calculate_mask(x_size).to(x.device))  # 计算注意力窗口并应用掩码

    # 合并窗口
    attn_windows = attn_windows.view(-1, self.window_size, self.window_size, c)  # 将注意力窗口调整为合适的形状
    shifted_x = window_reverse(attn_windows, self.window_size, h, w)  # 逆向恢复窗口

    # 逆向周期性偏移
    if self.shift_size > 0:
        x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))  # 逆向偏移
    else:
        x = shifted_x  # 若无偏移则直接使用
    x = x.view(b, h * w, c)  # 重新塑形返回到 (batch_size, height * width, channels)

    # 前馈网络
    x = shortcut + self.drop_path(x) + conv_x * self.cab_scale  # 加入卷积输出和残差连接
    x = x + self.drop_path(self.mlp(self.norm2(x)))  # 通过 MLP 前馈网络
    x = x + self.hin_scale * hin  # 加入 HIN 输出

    return x  # 返回最终输出


class PatchMerging(nn.Module):
    """Patch 合并层。

    参数：
        input_resolution (tuple[int]): 输入特征的分辨率。
        dim (int): 输入通道数。
        norm_layer (nn.Module, optional): 归一化层。默认: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):  # 初始化函数
        super().__init__()  # 初始化 nn.Module
        self.input_resolution = input_resolution  # 保存输入分辨率
        self.dim = dim  # 保存输入通道数
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)  # 合并后的线性层
        self.norm = norm_layer(4 * dim)  # 归一化层

    def forward(self, x):
        """
        参数：
            x: 输入特征 (b, h*w, c)

        返回：
            x: 合并后的特征
        """
        h, w = self.input_resolution  # 获取输入的高度和宽度
        b, seq_len, c = x.shape  # 获取输入的批次大小、序列长度和通道数
        assert seq_len == h * w, 'input feature has wrong size'  # 确保输入特征大小正确
        assert h % 2 == 0 and w % 2 == 0, f'x size ({h}*{w}) are not even.'  # 确保大小为偶数

        x = x.view(b, h, w, c)  # 重新塑形为 (batch_size, height, width, channels)

        # 划分并合并特征
        x0 = x[:, 0::2, 0::2, :]  # 获取 (0, 0) 位置的特征
        x1 = x[:, 1::2, 0::2, :]  # 获取 (1, 0) 位置的特征
        x2 = x[:, 0::2, 1::2, :]  # 获取 (0, 1) 位置的特征
        x3 = x[:, 1::2, 1::2, :]  # 获取 (1, 1) 位置的特征
        x = torch.cat([x0, x1, x2, x3], -1)  # 在最后一个维度上合并
        x = x.view(b, -1, 4 * c)  # 重新塑形为 (batch_size, height/2 * width/2, 4 * channels)

        x = self.norm(x)  # 归一化
        x = self.reduction(x)  # 线性变换

        return x  # 返回合并后的特征

    def extra_repr(self) -> str:  # 返回类的额外信息
        return f'input_resolution={self.input_resolution}, dim={self.dim}'

    def flops(self):
        """计算操作的 FLOPS"""
        h, w = self.input_resolution  # 获取输入特征的高和宽
        flops = h * w * self.dim  # 输入特征的 FLOPS
        flops += (h // 2) * (w // 2) * 4 * self.dim * 2 * self.dim  # 合并后的 FLOPS
        return flops  # 返回总 FLOPS


class BasicLayer(nn.Module):
    """基本的 Swin Transformer 层，用于一个阶段。

    参数：
        dim (int): 输入通道数。
        input_resolution (tuple[int]): 输入分辨率。
        depth (int): 块的数量。
        num_heads (int): 注意力头数量。
        window_size (int): 局部窗口大小。
        mlp_ratio (float): MLP 隐藏维度与嵌入维度的比例。
        qkv_bias (bool, optional): 如果为 True，则在查询、键、值中添加可学习的偏置。默认: True
        qk_scale (float | None, optional): 如果设置，则覆盖默认的 qk 缩放。
        drop (float, optional): 丢弃率。默认: 0.0
        attn_drop (float, optional): 注意力丢弃率。默认: 0.0
        drop_path (float | tuple[float], optional): 随机深度率。默认: 0.0
        norm_layer (nn.Module, optional): 归一化层。默认: nn.LayerNorm
        downsample (nn.Module | None, optional): 层末的下采样层。默认: None
        use_checkpoint (bool): 是否使用检查点保存内存。默认: False。
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 num_heads,
                 window_size,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False):
        super().__init__()  # 初始化 nn.Module
        self.dim = dim  # 保存通道数
        self.input_resolution = input_resolution  # 保存输入分辨率
        self.depth = depth  # 块的数量
        self.use_checkpoint = use_checkpoint  # 是否使用检查点

        # 创建块
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,  # 根据块的偶偶数设置偏移大小
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer) for i in range(depth)  # 创建多个 SwinTransformerBlock
        ])

        # 创建 Patch 合并层
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)  # 下采样模块
        else:
            self.downsample = None  # 无下采样

    def forward(self, x, x_size):  # 前向传播函数
        for blk in self.blocks:  # 遍历所有块
            if self.use_checkpoint:  # 如果使用检查点
                x = checkpoint.checkpoint(blk, x)  # 保存计算
            else:
                x = blk(x, x_size)  # 通过块处理输入

        if self.downsample is not None:  # 如果下采样模块存在
            x = self.downsample(x)  # 进行下采样
        return x  # 返回输出

    def extra_repr(self) -> str:  # 返回类的额外信息
        return f'dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}'

    def flops(self):
        """计算操作的 FLOPS"""
        flops = 0
        for blk in self.blocks:  # 遍历所有块计算 FLOPS
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()  # 加上下采样层的 FLOPS
        return flops  # 返回总 FLOPS


class RSTB(nn.Module):
    """残差 Swin Transformer 块（RSTB）。

    参数：
        dim (int): 输入通道数。
        input_resolution (tuple[int]): 输入分辨率。
        depth (int): 块的数量。
        num_heads (int): 注意力头数量。
        window_size (int): 局部窗口大小。
        mlp_ratio (float): MLP 隐藏维度与嵌入维度的比例。
        qkv_bias (bool, optional): 如果为 True，则在查询、键、值中添加可学习的偏置。默认: True
        qk_scale (float | None, optional): 如果设置，则覆盖默认的 qk 缩放。
        drop (float, optional): 丢弃率。默认: 0.0
        attn_drop (float, optional): 注意力丢弃率。默认: 0.0
        drop_path (float | tuple[float], optional): 随机深度率。默认: 0.0
        norm_layer (nn.Module, optional): 归一化层。默认: nn.LayerNorm
        downsample (nn.Module | None, optional): 层末的下采样层。默认: None
        use_checkpoint (bool): 是否使用检查点保存内存。默认: False。
        img_size: 输入图像大小。
        patch_size: patch 大小。
        resi_connection: 残差连接前的卷积块。
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 num_heads,
                 window_size,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 img_size=256,
                 patch_size=4,
                 resi_connection='1conv'):
        super(RSTB, self).__init__()  # 初始化 RSTB 类

        self.dim = dim  # 保存通道数
        self.input_resolution = input_resolution  # 保存输入分辨率

        self.residual_group = BasicLayer(  # 创建基础层
            dim=dim,
            input_resolution=input_resolution,
            depth=depth,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
            norm_layer=norm_layer,
            downsample=downsample,
            use_checkpoint=use_checkpoint)

        if resi_connection == '1conv':  # 如果需要残差连接
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)  # 创建卷积层

        self.patch_embed = PatchEmbed(  # 创建 Patch 嵌入层
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)

        self.patch_unembed = PatchUnEmbed(  # 创建 Patch 反嵌入层
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)

    def forward(self, x, x_size):
        """前向传播函数"""
        return self.patch_embed(self.conv(self.patch_unembed(self.residual_group(x, x_size), x_size))) + x  # 返回处理后的结果

    def flops(self):
        """计算操作的 FLOPS"""
        flops = 0
        flops += self.residual_group.flops()  # 加上残差组的 FLOPS
        h, w = self.input_resolution  # 获取输入分辨率
        flops += h * w * self.dim * self.dim * 9  # 加上卷积的 FLOPS
        flops += self.patch_embed.flops()  # 加上 Patch 嵌入的 FLOPS
        flops += self.patch_unembed.flops()  # 加上 Patch 反嵌入的 FLOPS

        return flops  # 返回总 FLOPS



class PatchEmbed(nn.Module):
    """图像到 Patch 嵌入。

    参数：
        img_size (int): 图像大小。默认: 256。
        patch_size (int): Patch token 大小。默认: 4。
        in_chans (int): 输入图像通道数。默认: 3。
        embed_dim (int): 线性投影输出通道数。默认: 96。
        norm_layer (nn.Module, optional): 归一化层。默认: None
    """

    def __init__(self, img_size=256, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()  # 初始化 nn.Module
        img_size = to_2tuple(img_size)  # 将输入图像大小转换为元组格式
        patch_size = to_2tuple(patch_size)  # 将输入 patch 大小转换为元组格式
        # 计算 patches 的分辨率（每个维度的 patch 数量）
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]  # 计算总 patch 数

        self.in_chans = in_chans  # 输入通道数
        self.embed_dim = embed_dim  # 嵌入维度

        # 如果提供了归一化层，则创建归一化层
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None  # 否则，不添加归一化

    def forward(self, x):
        # 将输入展平并转换维度
        x = x.flatten(2).transpose(1, 2)  # 变为 (batch, Ph*Pw, c)
        if self.norm is not None:  # 如果有归一化层
            x = self.norm(x)  # 对输入进行归一化
        return x  # 返回嵌入后的特征

    def flops(self):
        """计算 FLOPS"""
        flops = 0
        h, w = self.img_size  # 获取输入图像的高度和宽度
        if self.norm is not None:
            flops += h * w * self.embed_dim  # 如果有归一化，则加上归一化的 FLOPS
        return flops  # 返回总 FLOPS


class PatchUnEmbed(nn.Module):
    """图像到 Patch 反嵌入。

    参数：
        img_size (int): 图像大小。默认: 256。
        patch_size (int): Patch token 大小。默认: 4。
        in_chans (int): 输入图像通道数。默认: 3。
        embed_dim (int): 线性投影输出通道数。默认: 96。
        norm_layer (nn.Module, optional): 归一化层。默认: None
    """

    def __init__(self, img_size=256, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()  # 初始化 nn.Module
        img_size = to_2tuple(img_size)  # 将输入图像大小转换为元组格式
        patch_size = to_2tuple(patch_size)  # 将输入 patch 大小转换为元组格式
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans  # 输入通道数
        self.embed_dim = embed_dim  # 嵌入维度

    def forward(self, x, x_size):
        """前向传播函数。

        参数：
            x: 输入特征
            x_size: 输入特征的大小（高度和宽度）

        返回：
            x: 反嵌入后的特征
        """
        x = x.transpose(1, 2).contiguous().view(x.shape[0], self.embed_dim, x_size[0], x_size[1])  # 变换并重塑为 (b, c, h', w')
        return x  # 返回反嵌入后的特征

    def flops(self):
        """计算 FLOPS"""
        flops = 0
        return flops  # 返回 FLOPS


class Upsample(nn.Sequential):
    """上采样模块。

    参数：
        scale (int): 缩放因子。支持的缩放：2^n 和 3。
        num_feat (int): 中间特征的通道数。
    """

    def __init__(self, scale, num_feat):
        m = []  # 初始化模块列表
        if (scale & (scale - 1)) == 0:  # 如果 scale 是 2^n
            for _ in range(int(math.log(scale, 2))):  # 根据缩放因子的对数决定层数
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))  # 通过卷积扩大通道数
                m.append(nn.PixelShuffle(2))  # 使用 PixelShuffle 进行上采样
        elif scale == 3:  # 如果缩放因子为 3
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))  # 扩大通道数
            m.append(nn.PixelShuffle(3))  # 使用 PixelShuffle
        else:
            raise ValueError(f'scale {scale} is not supported. Supported scales: 2^n and 3.')  # 如果不是支持的缩放因子，抛出异常
        super(Upsample, self).__init__(*m)  # 调用父类的初始化


class Attention(nn.Module):
    """自校准注意力模块。

    参数：
        num_feat (int): 中间特征的通道数。
    """

    def __init__(self, nf):
        super(Attention, self).__init__()  # 初始化 nn.Module
        self.conv = nn.Sequential(  # 定义顺序容器
            nn.Conv2d(nf, nf, 1),  # 卷积层
            nn.ReLU(inplace=True),  # ReLU 激活函数
            nn.Conv2d(nf, nf, 1),  # 再次卷积
        )

    def forward(self, x):
        """前向传播函数。

        参数：
            x: 输入特征

        返回：
            out: 应用注意力后的输出
        """
        x1 = self.conv(x)  # 通过卷积层处理输入
        map_ = torch.sigmoid(x1)  # 应用 Sigmoid 激活生成注意力映射
        out = torch.mul(x1, map_)  # 进行通道加权
        return out  # 返回加权后的输出

# CIT模块
@ARCH_REGISTRY.register()
class MEC_CIT(nn.Module):
    """多曝光校正 CIT 模型。

    参数：
        nf (int): 中间特征通道数。
        depths (list[int]): 每一层的块数。
        num_heads (list[int]): 每一层的注意力头数量。
        window_size (int): 窗口大小。
        mlp_ratio (float): MLP 隐藏维度与嵌入维度的比例。
        qkv_bias (bool, optional): 是否为查询、键、值添加可学习偏置。默认: True
        norm_layer (nn.Module, optional): 归一化层。默认: nn.LayerNorm
        img_size (int): 输入图像的大小。默认: 256。
        patch_size (int): Patch 大小。默认: 1。
        resi_connection (str): 残差连接前的卷积块选择。默认: '1conv'
    """

    def __init__(self,
                 nf=180,
                 depths=[6, 6, 6, 6],
                 num_heads=[6, 6, 6, 6],
                 window_size=8,
                 mlp_ratio=2,
                 qkv_bias=True,
                 norm_layer=nn.LayerNorm,
                 img_size=256,
                 patch_size=1,
                 resi_connection='1conv'
                 ):
        super(MEC_CIT, self).__init__()  # 初始化 MEC_CIT 类

        self.conv_first = nn.Conv2d(3, nf, 4, 4)  # 首个卷积层，输入为 RGB 图像
        self.attention = Attention(nf)  # 创建注意力模块

        # 创建 Patch 嵌入层
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=nf,
            embed_dim=nf,
            norm_layer=norm_layer
        )
        patches_resolution = self.patch_embed.patches_resolution  # 获取 Patch 的分辨率

        self.layers = nn.ModuleList()  # 初始化模块列表
        for i_layer in range(len(depths)):  # 遍历每一层
            layer = RSTB(  # 创建 RSTB 块
                dim=nf,
                input_resolution=(patches_resolution[0], patches_resolution[1]),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                downsample=None,
                img_size=img_size,
                patch_size=patch_size,
                resi_connection=resi_connection
            )
            self.layers.append(layer)  # 将层添加到模块列表
        self.norm = norm_layer(nf)  # 最后层的归一化

        # 创建 Patch 反嵌入层
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=nf,
            embed_dim=nf,
            norm_layer=norm_layer
        )

        # 根据残差连接类型创建卷积层
        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2d(nf, nf, 3, 1, 1)

        # 创建上采样模块
        self.upsample = nn.Sequential(nn.Conv2d(nf, 3 * 4 ** 2, 3, 1, 1), nn.PixelShuffle(4))
        self.act = nn.GELU()  # 激活函数

        # 权重初始化
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """权重初始化函数"""
        if isinstance(m, nn.Linear):  # 初始化线性层的权重
            trunc_normal_(m.weight, std=.02)  # 截断正态分布初始化
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)  # 偏置初始化为 0
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)  # LayerNorm 的偏置初始化为 0
            nn.init.constant_(m.weight, 1.0)  # LayerNorm 的权重初始化为 1.0

    @torch.jit.ignore
    def no_weight_decay(self):
        """不进行权重衰减的参数"""
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        """不进行权重衰减的关键词"""
        return {'relative_position_bias_table'}

    def forward_feature(self, x):
        """提取特征的前向函数"""
        x_size = (x.shape[2], x.shape[3])  # 获取输入特征的大小
        x = self.patch_embed(x)  # 进行 Patch 嵌入

        for layer in self.layers:  # 通过每个层
            x = layer(x, x_size)  # 进行前向传播

        x = self.norm(x)  # 应用归一化
        x = self.patch_unembed(x, x_size)  # 进行 Patch 反嵌入

        return x  # 返回特征

    def forward(self, x):
        """前向传播函数."""
        # 获取输入图像的原始高度和宽度
        original_height = x.shape[2]
        original_width = x.shape[3]

        # 计算新的高度和宽度，确保符合窗口大小的要求
        new_height = ((original_height + 7) // 8) * 8  # 向上取整到下一个能被8整除的数
        new_width = ((original_width + 7) // 8) * 8  # 向上取整到下一个能被8整除的数

        # print(f"Resizing to: height={new_height}, width={new_width}")

        # 使用 F.interpolate 调整输入图像大小
        x = F.interpolate(x, size=(new_height, new_width), mode='bilinear', align_corners=False)

        """前向传播函数."""
        x = F.relu(self.conv_first(x))  # 通过第一个卷积层并进行 ReLU 激活
        res = x  # 缓存快捷连接
        x = self.attention(x)  # 通过注意力模块
        x = self.conv_after_body(self.forward_feature(x)) + res  # 反向特征并进行残差连接
        out = F.relu(self.upsample(x))  # 进行上采样并激活
        return out  # 返回最终输出



