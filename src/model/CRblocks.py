import torch  # 导入PyTorch库
import torch.nn as nn  # 导入PyTorch的神经网络模块
import torch.nn.functional as F  # 导入PyTorch的功能性接口

# --------------------------------
# Convolutional Enhancement Block卷积增强模块
# --------------------------------
# 定义一个反转块类
class invertedBlock(nn.Module):
    def __init__(self, in_channel, out_channel, ratio=2):
        super(invertedBlock, self).__init__()  # 初始化nn.Module
        internal_channel = in_channel * ratio  # 计算内部通道数
        self.relu = nn.GELU()  # 使用GELU激活函数
        # 并行的7x7卷积与3x3卷积
        self.conv1 = nn.Conv2d(internal_channel, internal_channel, 7, 1, 3, groups=in_channel, bias=False)  # 7x7卷积

        self.convFFN = ConvFFN(in_channels=in_channel, out_channels=in_channel)  # 定义前馈网络
        self.layer_norm = nn.LayerNorm(in_channel)  # 定义层归一化
        # 通道扩展的1x1卷积
        self.pw1 = nn.Conv2d(in_channels=in_channel, out_channels=internal_channel, kernel_size=1, stride=1,
                             padding=0, groups=1, bias=False)
        # 通道收缩的1x1卷积
        self.pw2 = nn.Conv2d(in_channels=internal_channel, out_channels=in_channel, kernel_size=1, stride=1,
                             padding=0, groups=1, bias=False)

    # 高级功能
    def hifi(self, x):
        x1 = self.pw1(x)  # 通过第一层1x1卷积
        x1 = self.relu(x1)  # 激活
        x1 = self.conv1(x1)  # 7x7卷积
        x1 = self.relu(x1)  # 再次激活
        x1 = self.pw2(x1)  # 通过第二层1x1卷积
        x1 = self.relu(x1)  # 激活
        x3 = x1 + x  # 残差连接

        x3 = x3.permute(0, 2, 3, 1).contiguous()  # 转换张量维度
        x3 = self.layer_norm(x3)  # 进行层归一化
        x3 = x3.permute(0, 3, 1, 2).contiguous()  # 转换为原始维度
        x4 = self.convFFN(x3)  # 传入前馈网络

        return x4  # 返回结果

    # 定义前向传播
    def forward(self, x):
        return self.hifi(x) + x  # 返回经过处理后的输出加上输入（残差连接）

# 定义前馈网络类
class ConvFFN(nn.Module):
    def __init__(self, in_channels, out_channels, expend_ratio=4):
        super().__init__()  # 初始化nn.Module

        internal_channels = in_channels * expend_ratio  # 计算内部通道数
        self.pw1 = nn.Conv2d(in_channels=in_channels, out_channels=internal_channels, kernel_size=1, stride=1,
                             padding=0, groups=1, bias=False)  # 1x1卷积
        self.pw2 = nn.Conv2d(in_channels=internal_channels, out_channels=out_channels, kernel_size=1, stride=1,
                             padding=0, groups=1, bias=False)  # 1x1卷积
        self.nonlinear = nn.GELU()  # 使用GELU激活函数

    # 定义前向传播
    def forward(self, x):
        x1 = self.pw1(x)  # 通过第一层1x1卷积
        x2 = self.nonlinear(x1)  # 激活
        x3 = self.pw2(x2)  # 通过第二层1x1卷积
        x4 = self.nonlinear(x3)  # 激活
        return x4 + x  # 返回经过处理后的输出加上输入（残差连接）



# --------------------------------
# Frequence Separation and Fusion 频域的分离与融合
# --------------------------------
# 定义混合块类MBB
class mixblock(nn.Module):
    def __init__(self, n_feats):
        super(mixblock, self).__init__()  # 初始化nn.Module
        # 定义两个卷积链
        self.conv1 = nn.Sequential(nn.Conv2d(n_feats, n_feats, 3, 1, 1, bias=False), nn.GELU())  # 第一卷积链1x1
        self.conv2 = nn.Sequential(nn.Conv2d(n_feats, n_feats, 3, 1, 1, bias=False),
                                   nn.GELU(),
                                   nn.Conv2d(n_feats, n_feats, 3, 1, 1, bias=False),
                                   nn.GELU(),
                                   nn.Conv2d(n_feats, n_feats, 3, 1, 1, bias=False),
                                   nn.GELU())  # 第二卷积链3x3

        self.alpha = nn.Parameter(torch.ones(1))  # 定义可学习参数
        self.beta = nn.Parameter(torch.ones(1))  # 定义可学习参数

    # 定义前向传播
    def forward(self, x):
        return self.alpha * self.conv1(x) + self.beta * self.conv2(x)  # 返回两个卷积的加权和

# 通道注意力层
class CALayer(nn.Module):
    def __init__(self, channel, reduction=2):
        super(CALayer, self).__init__()  # 初始化nn.Module
        # 全局平均池化：特征图 --> 点
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 自适应平均池化
        # 特征通道降维和升维 --> 通道权重
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),  # 降维
            nn.ReLU(inplace=True),  # 激活函数
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),  # 升维
            nn.Sigmoid()  # Sigmoid激活，以生成权重
        )

    # 定义前向传播
    def forward(self, x):
        # print(f'Input shape to CALayer: {x.shape}')
        # assert x.size(1) > 0, "Channel size should be greater than 0."
        y = self.avg_pool(x)  # 计算全局平均池化
        y = self.conv_du(y)  # 通过卷积计算权重
        return x * y  # 返回输入特征与权重的逐通道乘积

# 向下与向上块
class Downupblock(nn.Module):
    def __init__(self, n_feats):
        super(Downupblock, self).__init__()  # 初始化nn.Module
        self.encoder = mixblock(n_feats)  # 定义编码器
        self.decoder_high = mixblock(n_feats)  # 定义高频解码器
        self.decoder_low = nn.Sequential(mixblock(n_feats), mixblock(n_feats), mixblock(n_feats))  # 定义低频解码器
        self.alise = nn.Conv2d(n_feats, n_feats, 1, 1, 0, bias=False)  # 定义1x1卷积层
        self.alise2 = nn.Conv2d(n_feats * 2, n_feats, 3, 1, 1, bias=False)  # 定义3x3卷积层
        self.down = nn.AvgPool2d(kernel_size=2)  # 定义平均池化层
        self.att = CALayer(n_feats)  # 定义通道注意力层
        self.raw_alpha = nn.Parameter(torch.ones(1))  # 定义可学习参数，用于加权

        self.raw_alpha.data.fill_(0)  # 初始化参数为0
        self.ega = selfAttention(n_feats, n_feats)  # 定义自注意力层

    # 定义前向传播
    def forward(self, x, raw):
        x1 = self.encoder(x)  # 编码输入
        x2 = self.down(x1)  # 下采样
        high = x1 - F.interpolate(x2, size=x.size()[-2:], mode='bilinear', align_corners=True)  # 计算高频组件

        high = high + self.ega(high, high) * self.raw_alpha  # 加入自注意力
        x2 = self.decoder_low(x2)  # 解码低频
        x3 = x2  # 保存低频组件
        high1 = self.decoder_high(high)  # 解码高频
        x4 = F.interpolate(x3, size=x.size()[-2:], mode='bilinear', align_corners=True)  # 低频上采样
        return self.alise(self.att(self.alise2(torch.cat([x4, high1], dim=1)))) + x  # 输出结合

# 向上与向下块
class Updownblock(nn.Module):
    def __init__(self, n_feats):
        super(Updownblock, self).__init__()  # 初始化nn.Module
        self.encoder = mixblock(n_feats)  # 定义编码器
        self.decoder_high = mixblock(n_feats)  # 定义高频解码器
        self.decoder_low = nn.Sequential(mixblock(n_feats), mixblock(n_feats), mixblock(n_feats))  # 定义低频解码器

        self.alise = nn.Conv2d(n_feats, n_feats, 1, 1, 0, bias=False)  # 定义1x1卷积层
        self.alise2 = nn.Conv2d(n_feats * 2, n_feats, 3, 1, 1, bias=False)  # 定义3x3卷积层
        self.down = nn.AvgPool2d(kernel_size=2)  # 定义平均池化层
        self.att = CALayer(n_feats)  # 定义通道注意力层
        self.raw_alpha = nn.Parameter(torch.ones(1))  # 定义可学习参数

        self.raw_alpha.data.fill_(0)  # 初始化参数为0
        self.ega = selfAttention(n_feats, n_feats)  # 定义自注意力层

    # 定义前向传播
    def forward(self, x, raw):
        x1 = self.encoder(x)  # 编码输入
        x2 = self.down(x1)  # 下采样
        high = x1 - F.interpolate(x2, size=x.size()[-2:], mode='bilinear', align_corners=True)  # 计算高频组件
        high = high + self.ega(high, high) * self.raw_alpha  # 加入自注意力
        x2 = self.decoder_low(x2)  # 解码低频
        x3 = x2  # 保存低频组件
        high1 = self.decoder_high(high)  # 解码高频
        x4 = F.interpolate(x3, size=x.size()[-2:], mode='bilinear', align_corners=True)  # 上采样
        return self.alise(self.att(self.alise2(torch.cat([x4, high1], dim=1)))) + x  # 输出结合

# 定义基本块
class basic_block(nn.Module):
    ## 双并行分支，通道分支和空间分支
    def __init__(self, in_channel=3, out_channel=3, depth=1, ratio=1):
        super(basic_block, self).__init__()  # 初始化nn.Module
        print(f'Input channels: {in_channel}, Output channels: {out_channel}')
        # 重复定义的深度为depth的块
        self.rep1 = nn.Sequential(*[invertedBlock(in_channel=in_channel, out_channel=in_channel, ratio=ratio) for i in range(depth)])  # 深度重复模块

        self.relu = nn.ReLU(inplace=True)  # 使用ReLU激活函数
        # 向上和向下块
        self.updown = Updownblock(in_channel)  # 定义向上与向下块
        self.downup = Downupblock(in_channel)  # 定义向下与向上块

    # 定义前向传播
    def forward(self, x, raw=None):
        x1 = self.rep1(x)  # 多次应用反转块

        x1 = self.updown(x1, raw)  # 应用向上与向下块
        x1 = self.downup(x1, raw)  # 应用向下与向上块
        x2 = x1 + x  # 残差连接
        out = torch.clamp(x2, 0, 1)
        return out  # 返回输出与输入的和

import torchvision  # 导入torchvision库

# 定义VGG感知模块
class VGG_aware(nn.Module):
    def __init__(self, outFeature):
        super(VGG_aware, self).__init__()  # 初始化nn.Module
        blocks = []  # 初始化模块列表
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())  # 提取VGG16的前4层特征

        for bl in blocks:  # 冻结所有层的参数
            for p in bl:
                p.requires_grad = False  # 不进行梯度更新
        self.blocks = torch.nn.ModuleList(blocks)  # 保存为nn.ModuleList

    # 定义前向传播
    def forward(self, x):
        return self.blocks[0](x)  # 返回提取的特征

import torch.nn.functional as f  # 导入PyTorch功能性接口

# 定义自注意力层
class selfAttention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(selfAttention, self).__init__()  # 初始化nn.Module
        self.query_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)  # 查询卷积
        self.key_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)  # 键卷积
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)  # 值卷积
        self.scale = 1.0 / (out_channels ** 0.5)  # 缩放因子

    # 定义前向传播
    def forward(self, feature, feature_map):
        query = self.query_conv(feature)  # 计算查询
        key = self.key_conv(feature)  # 计算键
        value = self.value_conv(feature)  # 计算值
        attention_scores = torch.matmul(query, key.transpose(-2, -1))  # 计算注意力得分
        attention_scores = attention_scores * self.scale  # 进行缩放

        attention_weights = f.softmax(attention_scores, dim=-1)  # 计算注意力权重

        attended_values = torch.matmul(attention_weights, value)  # 应用权重计算最终值

        output_feature_map = (feature_map + attended_values)  # 输出图特征图

        return output_feature_map  # 返回更新后的特征图
