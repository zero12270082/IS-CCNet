import pdb
import sys

import torch
import torch.nn.functional as F
from torch import nn

# ACCM模块
class ColorDeformConv2d(nn.Module):
    def __init__(
        self,
        inc,# 输入通道数
        outc,  # 输出通道数
        kernel_size=3,  # 卷积核大小，默认为3
        padding=1,  # 填充大小，默认为1
        stride=1,  # 卷积步长，默认为1
        bias=None, # 是否添加偏置项，默认为None，即根据nn.Conv2d的默认行为决定
        modulation=False,  # 是否使用调制变形卷积（Deformable ConvNets v2），默认为False
        color_deform=True,  # 是否使用颜色变形卷积，默认为True
    ):
        """
        Args:
            modulation (bool, optional): If True, Modulated Defomable Convolution (Deformable ConvNets v2).
        """
        super(ColorDeformConv2d, self).__init__()
        # 保存卷积核大小、填充大小和步长等参数
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        # 使用ZeroPad2d层对输入进行填充，填充大小为padding
        self.zero_padding = nn.ZeroPad2d(padding)

        # 定义一个标准的卷积层，用于初步的特征提取

        # self.conv = nn.Conv2d(
        #     inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias
        # )
        self.conv = nn.Conv2d(
            inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias
        )
        # 定义一个1x1卷积层，用于通道数下采样，将输入通道数减半（假设输入通道数为偶数）
        # 注意这里输入通道数是inc * 2
        # self.channel_down = nn.Conv2d(inc * 2, inc, kernel_size=1, stride=1)
        self.channel_down = nn.Conv2d(inc * 2, inc, kernel_size=1, stride=1)
        # 定义一个3x3卷积层，用于预测每个位置上的偏移量
        # 输出的通道数为2 * kernel_size * kernel_size，对应每个卷积核位置的x和y偏移量
        #位置偏移 Δpn
        self.p_conv = nn.Conv2d(
            inc,
            2 * kernel_size * kernel_size,
            kernel_size=3,
            padding=1,
            stride=stride,
        )
        # 判断是否使用调制变形卷积
        self.modulation = modulation
        if modulation:
            # 如果使用调制变形卷积，定义一个额外的3x3卷积层，用于预测每个位置上的调制权重
            # 输出的通道数为kernel_size * kernel_size，对应每个卷积核位置的调制权重
            # 调制项 Δmn
            self.m_conv = nn.Conv2d(
                inc,
                kernel_size * kernel_size,
                kernel_size=3,
                padding=1,
                stride=stride,
            )
        # 判断是否使用颜色变形卷积
        self.color_deform = color_deform
        if self.color_deform:
            # 如果使用颜色变形卷积，定义一个3x3卷积层，用于预测每个位置上的颜色变换参数
            # 输出的通道数为kernel_size * kernel_size * inc，对应每个卷积核位置和每个输入通道的颜色变换参数
            #颜色偏移量 Δcn
            self.c_conv = nn.Conv2d(
                inc,
                kernel_size * kernel_size * inc,
                kernel_size=3,
                padding=1,
                stride=stride,
            )

    def forward(self, x, ref):
        # 清理CUDA缓存
        torch.cuda.empty_cache()
        # 断言输入x和参考图像ref的形状相同
        assert (
            x.shape == ref.shape
        ), f"Input shape {x.shape} and reference shape {ref.shape} does not match."
        # 获取输入x的批次大小、通道数、高度和宽度
        b, c, h, w = x.size()
        # 将输入x和参考图像ref在通道维度上拼接
        fused = torch.cat([x, ref], dim=1)  # (b, 2c, h, w)
        # 使用1x1卷积层对拼接后的特征图进行通道下采样，恢复到原始通道数c
        fused = self.channel_down(fused)  # (b, c, h, w)
        # 断言融合后的特征图形状与输入x的形状相同
        assert (
            fused.shape == x.shape
        ), f"Fused shape {fused.shape} and input shape {x.shape} does not match."
        # 使用3x3卷积层预测每个位置上的偏移量--位置偏移量 Δpn
        offset = self.p_conv(fused)  # (b, c, h, w) -> (b, 2*ks*ks, h, w)
        # 如果启用了调制变形卷积
        if self.modulation:
            # 使用sigmoid函数将调制权重限制在0到1之间
            m = torch.sigmoid(
                self.m_conv(fused)
            )  # (b, c, h, w) -> (b, ks*ks, h, w)
            # 如果启用了颜色变形卷积
        if self.color_deform:
            # 使用tanh函数将颜色变换参数限制在-1到1之间
            c_offset = torch.tanh(
                self.c_conv(fused)
            )  # (b, c, h, w)->(b, c*ks*ks, h, w)
        # 获取偏移量的数据类型
        dtype = offset.data.type()
        # 获取卷积核大小
        ks = self.kernel_size  # ks = 3
        # 计算卷积核大小的平方，即每个位置需要的采样点数
        N = ks * ks  # N = kernel_size * kernel_size = 9
        # 如果启用了填充，则对输入x进行填充
        if self.padding:  # self.padding = 1
            x = self.zero_padding(
                x
            )  # x.shape: (b, c, h, w) -> (b, c, h+2, w+2)

        # (b, 2N, h, w)---第一个
        # 从偏移量中获取采样点的位置坐标，并进行必要的调整
        p = self._get_p(offset, dtype)  # offset.shape: (1, 18, h, w)

        # (b, h, w, 2N)
        # 调整p的形状和维度，以便后续计算
        p = p.contiguous().permute(0, 2, 3, 1)
        # 计算四个角点的坐标（左上、右上、左下、右下）
        q_lt = p.detach().floor()  # left top
        q_rb = q_lt + 1  # right bottom
        # 对坐标进行裁剪，确保它们不会超出输入x的边界
        q_lt = torch.cat(
            [
                torch.clamp(q_lt[..., :N], 0, x.size(2) - 1),
                torch.clamp(q_lt[..., N:], 0, x.size(3) - 1),
            ],
            dim=-1,
        ).long()
        q_rb = torch.cat(
            [
                torch.clamp(q_rb[..., :N], 0, x.size(2) - 1),
                torch.clamp(q_rb[..., N:], 0, x.size(3) - 1),
            ],
            dim=-1,
        ).long()
        # 计算左下和右上角点的坐标
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

        # clip p
        # 对p进行裁剪，确保它们不会超出输入x的边界
        p = torch.cat(
            [
                torch.clamp(p[..., :N], 0, x.size(2) - 1),
                torch.clamp(p[..., N:], 0, x.size(3) - 1),
            ],
            dim=-1,
        )

        # bilinear kernel (b, h, w, N)
        # 计算双线性插值的权重
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (
            1 + (q_lt[..., N:].type_as(p) - p[..., N:])
        )
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (
            1 - (q_rb[..., N:].type_as(p) - p[..., N:])
        )
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (
            1 - (q_lb[..., N:].type_as(p) - p[..., N:])
        )
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (
            1 + (q_rt[..., N:].type_as(p) - p[..., N:])
        )


        # (b, c, h, w, N)
        # 根据计算出的坐标和权重，从输入x中采样得到四个角点的值
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # (b, c, h, w, ks*ks)
        # 使用双线性插值计算得到最终的采样值
        x_offset = (
            g_lt.unsqueeze(dim=1) * x_q_lt
            + g_rb.unsqueeze(dim=1) * x_q_rb
            + g_lb.unsqueeze(dim=1) * x_q_lb
            + g_rt.unsqueeze(dim=1) * x_q_rt
        )
        # 如果启用了颜色变形卷积，则将颜色变换参数加到采样值上
        if self.color_deform:
            c_offset = c_offset.contiguous().reshape(
                b, c, N, h, w
            )  # (b, c*ks*ks, h, w) -> (b, c, ks*ks, h, w)
            c_offset = c_offset.contiguous().permute(
                0, 1, 3, 4, 2
            )  # (b, c, ks*ks, h, w) -> (b, c, h, w, ks*ks)
            x_offset += c_offset

        # modulation
        # 如果启用了调制变形卷积，则使用调制权重对采样值进行调制
        if self.modulation:
            m = m.contiguous().permute(0, 2, 3, 1)  # (b, h, w, ks*ks)
            m = m.unsqueeze(dim=1)  # (b, 1, h, w, ks*ks)
            m = torch.cat(
                [m for _ in range(x_offset.size(1))], dim=1
            )  # (b, 1, h, w, ks*ks) -> (b, c, h, w, ks*ks)
            x_offset *= m
        # 将x_offset的形状调整为适合标准卷积输入的形状
        x_offset = self._reshape_x_offset(x_offset, ks)  # (b, c, h*ks, w*ks)
        # 使用标准卷积层对调整形状后的特征图进行卷积操作
        out = self.conv(x_offset)
        # 返回卷积操作的输出
        return out

    # 定义一个私有方法，用于生成相对于卷积核中心点的偏移坐标网格
    def _get_p_n(self, N, dtype):
        # 使用torch.meshgrid生成一个二维网格，网格的范围是卷积核大小的一半（上下左右）
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(
                -(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1
            ),
            torch.arange(
                -(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1
            ),
        )
        # (2N, 1)
        # 将x和y坐标展平并拼接起来，形成一个长度为2N的一维向量
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
        # 将这个向量重新塑形为(1, 2N, 1, 1)的形状，并转换为指定的数据类型
        p_n = p_n.view(1, 2 * N, 1, 1).type(dtype)

        return p_n

    # 定义一个私有方法，用于生成采样点的初始位置（即特征图上的点）
    def _get_p_0(self, h, w, N, dtype):
        # 使用torch.meshgrid生成一个二维网格，网格的步长是stride，范围覆盖整个特征图
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(1, h * self.stride + 1, self.stride),
            torch.arange(1, w * self.stride + 1, self.stride),
        )
        # 将x和y坐标展平并重新塑形为(1, 1, h, w)的形状，然后沿着N的维度复制，形成(1, N, h, w)的形状
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        # 将x和y坐标拼接起来，形成一个(1, 2N, h, w)的形状，并转换为指定的数据类型
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)

        return p_0

    # 定义一个私有方法，用于计算最终的采样点位置
    def _get_p(self, offset, dtype):
        # 从offset中提取出N, h, w的值
        N, h, w = offset.size(1) // 2, offset.size(2), offset.size(3)

        # (1, 2N, 1, 1)
        # 调用_get_p_n方法获取相对于卷积核中心点的偏移坐标网格
        p_n = self._get_p_n(N, dtype)
        # (1, 2N, h, w)
        # 调用_get_p_0方法获取采样点的初始位置
        p_0 = self._get_p_0(h, w, N, dtype)
        # 将初始位置、偏移网格和额外的offset相加，得到最终的采样点位置
        p = p_0 + p_n + offset
        return p

    # 定义一个私有方法，用于根据采样点位置q从输入特征图x中采样
    def _get_x_q(self, x, q, N):
        # 从q中提取出批次大小b、高度h、宽度w
        b, h, w, _ = q.size()
        # 获取输入特征图x的宽度（可能经过padding后的宽度）
        padded_w = x.size(3)
        # 获取输入特征图的通道数c
        c = x.size(1)
        # (b, c, h*w)
        # 将输入特征图x重塑为(b, c, h*w)的形状，以便后续操作
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, N)
        # 从q中提取出采样点的偏移量（分为x方向和y方向），并计算采样点在输入特征图上的绝对位置
        index = q[..., :N] * padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        # 将index重塑并扩展维度，以便与x进行gather操作
        index = (
            index.contiguous()
            .unsqueeze(dim=1)
            .expand(-1, c, -1, -1, -1)
            .contiguous()
            .view(b, c, -1)
        ).type(torch.int64)
        # 使用gather操作根据index从x中采样，得到采样后的特征图x_offset
        x_offset = (
            x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)
        )

        return x_offset

    # 定义一个静态私有方法，用于将采样后的特征图x_offset重新塑形
    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        # 从x_offset中提取出批次大小b、通道数c、高度h、宽度w和采样点数N
        b, c, h, w, N = x_offset.size()
        # 将x_offset按照ks（卷积核大小）进行分组，并重新塑形为(b, c, h*ks, w*ks)的形状
        x_offset = torch.cat(
            [
                x_offset[..., s : s + ks].contiguous().view(b, c, h, w * ks)
                for s in range(0, N, ks)
            ],
            dim=-1,
        )
        # 将x_offset进一步重塑为(b, c, h*ks, w*ks)的形状
        x_offset = x_offset.contiguous().view(b, c, h * ks, w * ks)

        return x_offset
