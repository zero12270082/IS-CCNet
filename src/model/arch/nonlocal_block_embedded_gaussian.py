import ipdb
import torch
from torch import nn
from torch.nn import functional as F


class _NonLocalBlockND(nn.Module):
    def __init__(
        self,
        in_channels, # 输入通道数
        inter_channels=None, # 中间通道数，如果未指定则默认为输入通道数的一半（但至少为1）
        dimension=3,  # 数据维度，支持1D、2D和3D
        sub_sample="pool",  # 下采样方法，支持'pool'（池化）、'bilinear'（双线性插值）或False（不进行下采样）
        bn_layer=True,  # 是否在输出卷积后添加批量归一化层
    ):
        """
        :param in_channels:
        :param inter_channels:
        :param dimension:
        :param sub_sample: 'pool' or 'bilinear' or False
        :param bn_layer:
        """

        super(_NonLocalBlockND, self).__init__()
        # 断言维度参数必须在[1, 2, 3]中
        assert dimension in [1, 2, 3]
        # 初始化类属性
        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels
        # 如果中间通道数未指定，则设置为输入通道数的一半，但不得小于1
        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1
        # 根据数据维度选择相应的卷积层、池化层和批量归一化层
        if dimension == 3:
            conv_nd = nn.Conv3d # 3D卷积
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))  # 3D最大池化
            bn = nn.BatchNorm3d  # 3D批量归一化
        elif dimension == 2:
            conv_nd = nn.Conv2d  # 2D卷积
            # 根据下采样方法选择相应的层
            if sub_sample == "pool":
                max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2)) # 2D最大池化
            elif sub_sample == "bilinear":
                max_pool_layer = nn.UpsamplingBilinear2d([16, 16])# 2D双线性上采样
            else:
                raise NotImplementedError(
                    f"[ ERR ] Unknown down sample method: {sub_sample}"
                )
            bn = nn.BatchNorm2d # 2D批量归一化
        else:
            conv_nd = nn.Conv1d  # 1D卷积
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))  # 1D最大池化
            bn = nn.BatchNorm1d # 1D批量归一化
        # 定义g函数，用于将输入特征映射到中间通道数
        self.g = conv_nd(
            in_channels=self.in_channels,
            out_channels=self.inter_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        # 根据是否添加批量归一化层定义W函数，用于将中间特征映射回输入通道数
        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(
                    in_channels=self.inter_channels,
                    out_channels=self.in_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                ),
                bn(self.in_channels), # 添加批量归一化层
            )
            # 初始化批量归一化层的权重和偏置为0
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(
                in_channels=self.inter_channels,
                out_channels=self.in_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            )
            # 初始化权重和偏置为0
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)
        # 定义theta和phi函数，用于计算非局部注意力机制的键和查询
        self.theta = conv_nd(
            in_channels=self.in_channels,
            out_channels=self.inter_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.phi = conv_nd(
            in_channels=self.in_channels,
            out_channels=self.inter_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        # 如果启用了下采样，则将g和phi的输出通过下采样层
        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    # 定义一个名为forward的函数，它是模型的前向传播函数。这个函数接受输入x和一个可选参数return_nl_map。
    # 参数x的形状为(b, c, t, h, w)，分别代表批次大小、通道数、时间维度（或序列长度）、高度和宽度。
    # 参数return_nl_map是一个布尔值，如果为True，则函数返回z和nl_map（非局部映射），否则只返回z。
    def forward(self, x, return_nl_map=False):
        """
        :param x: (b, c, t, h, w)
        :param return_nl_map: if True return z, nl_map, else only return z.
        :return:
        """
        # 获取输入x的批次大小
        batch_size = x.size(0)
        # 通过函数g处理输入x，并将结果重塑为(batch_size, self.inter_channels, -1)
        # 其中self.inter_channels是中间通道数，-1表示自动计算该维度的大小。
        # 然后将结果的维度从(batch_size, self.inter_channels, -1)置换为(batch_size, -1, self.inter_channels)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        # 通过函数theta处理输入x，并进行与g_x相同的重塑和置换操作。
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        # 通过函数phi处理输入x，并进行与g_x相同的重塑操作，但不进行置换。
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        # 计算theta_x和phi_x的矩阵乘法，得到f。
        # f的每个元素表示不同位置间的相似度或关系。
        f = torch.matmul(theta_x, phi_x)
        # 对f应用softmax函数，得到f_div_C。
        # 这一步是为了将f的每一行归一化为概率分布，以便后续计算加权和。
        f_div_C = F.softmax(f, dim=-1)

        # 使用f_div_C作为权重，对g_x进行加权求和，得到y。
        # 这相当于根据位置间的相似度对g_x进行非局部聚合。
        y = torch.matmul(f_div_C, g_x)
        # 将y的维度从(batch_size, -1, self.inter_channels)置换回(batch_size, self.inter_channels, -1)
        # 并确保y在内存中是连续的。
        y = y.permute(0, 2, 1).contiguous()
        # 将y重塑为与输入x相同的形状(batch_size, self.inter_channels, t, h, w)。
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        # 通过函数W处理y，得到W_y。
        W_y = self.W(y)
        # 将W_y与输入x相加，得到最终的输出z。
        # 这一步是将非局部信息融合回原始输入中
        z = W_y + x
        # 如果return_nl_map为True，则返回z和f_div_C（非局部映射）。
        # f_div_C表示输入特征图中不同位置间的相似度或关系映射。
        if return_nl_map:
            return z, f_div_C
        # 否则，只返回z
        return z

# 实现了一维非局部神经网络模块
class NONLocalBlock1D(_NonLocalBlockND):
    def __init__(
            # 类的初始化方法，接受输入通道数in_channels、中间通道数inter_channels（可选）、
            # 下采样方式sub_sample（默认为"pool"）、以及是否使用批归一化层bn_layer（默认为True）。
            self, in_channels, inter_channels=None, sub_sample="pool", bn_layer=True
    ):
        # 调用基类的初始化方法，传入输入通道数、中间通道数（如果未指定，则使用基类中的逻辑处理）、
        # 维度（对于一维非局部模块，维度为1）、下采样方式、以及是否使用批归一化层。
        super(NONLocalBlock1D, self).__init__(
            in_channels,#输入特征图的通道数
            inter_channels=inter_channels,#中间通道数，用于减少计算量。如果未指定，则可能在基类中使用某种策略来确定
            dimension=1,
            sub_sample=sub_sample,
            bn_layer=bn_layer,#一个布尔值，指示是否在模块中使用批归一化层
        )


class NONLocalBlock2D(_NonLocalBlockND):
    # 类的初始化方法与NONLocalBlock1D类似，但维度参数设置为2，以表示二维非局部模块。
    def __init__(
        self, in_channels, inter_channels=None, sub_sample="pool", bn_layer=True
    ):
        super(NONLocalBlock2D, self).__init__(
            in_channels,
            inter_channels=inter_channels,
            dimension=2,
            sub_sample=sub_sample,
            bn_layer=bn_layer,
        )


class NONLocalBlock3D(_NonLocalBlockND):
    # 类的初始化方法与NONLocalBlock1D和NONLocalBlock2D类似，但维度参数设置为3，以表示三维非局部模块。
    def __init__(
        self, in_channels, inter_channels=None, sub_sample="pool", bn_layer=True
    ):
        super(NONLocalBlock3D, self).__init__(
            in_channels,
            inter_channels=inter_channels,
            dimension=3,
            sub_sample=sub_sample,
            bn_layer=bn_layer,
        )
