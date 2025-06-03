import os
from collections import OrderedDict
from .CTB import MEC_CIT
from .ODC import ODConv2d
from .hwd import Down_wt
import os
import torchvision.utils
import sys
import torchvision.transforms.functional as TF
from .bilateralupsamplenet import BilateralUpsampleNet
from torch.nn import DataParallel
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(project_root))
# from model import parse_model_class
sys.path.insert(0, '../')
import src.utils.util as util
from src.globalenv import *
from .arch.cross_nonlocal import CrossNonLocalBlock
from .arch.nonlocal_block_embedded_gaussian import NONLocalBlock2D
from .basic_loss import *
from .dconv import ColorDeformConv2d
from .single_net_basemodel import SingleNetBaseModel
# from RAM_lut import MySampler,SepLUT

class LayerNormFunction(torch.autograd.Function):
    """
    层归一化函数的实现。

    层归一化是一种用于神经网络的归一化技术，它旨在规范化每个特征通道的均值和方差。
    这个类包含了前向传播和后向传播的静态方法。

    参数:
    - x: 输入张量。
    - weight: 权重张量，用于缩放归一化后的张量。
    - bias: 偏置张量，用于偏移归一化后的张量。
    - eps: 一个小的正数，用于避免除以零的错误。
    """

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        """
        层归一化的前向传播过程。

        参数:
        - ctx: 上下文对象，用于存储用于反向传播的信息。
        - x: 输入张量。
        - weight: 权重张量。
        - bias: 偏置张量。
        - eps: 避免除以零的epsilon值。

        返回:
        - y: 经过层归一化的输出张量。
        """
        ctx.eps = eps
        # 计算输入张量的形状
        N, C, H, W = x.size()
        # 计算每个特征通道的均值
        mu = x.mean(1, keepdim=True)
        # 计算每个特征通道的方差
        var = (x - mu).pow(2).mean(1, keepdim=True)
        # 归一化输入张量
        y = (x - mu) / (var + eps).sqrt()
        # 保存归一化后的张量和方差，以及权重，用于后向传播
        ctx.save_for_backward(y, var, weight)
        # print(f"Input shape: {x.shape}")
        # print(f"Weight shape: {weight.shape}")
        # print(f"Bias shape: {bias.shape}")
        # 应用权重和偏置
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        """
        层归一化的后向传播过程。

        参数:
        - ctx: 上下文对象，存储了前向传播的信息。
        - grad_output: 输入张量的梯度。

        返回:
        - 输入张量的梯度。
        - 权重张量的梯度。
        - 偏置张量的梯度。
        - None，表示没有需要额外存储的参数。
        """
        eps = ctx.eps
        # 获取输入张量的形状
        N, C, H, W = grad_output.size()
        # 从上下文对象中获取保存的张量
        y, var, weight = ctx.saved_tensors
        # 将权重应用到梯度输出上
        g = grad_output * weight.view(1, C, 1, 1)
        # 计算每个特征通道的平均梯度
        mean_g = g.mean(dim=1, keepdim=True)
        # 计算用于更新输入张量的梯度
        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        # 计算权重和偏置的梯度，然后返回
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None

class LayerNorm2d(nn.Module):
    """
    二维层归一化模块。

    该模块对二维张量进行层归一化处理。层归一化旨在解决循环神经网络中梯度消失和爆炸的问题，
    通过归一化每个特征通道的均值和方差来稳定训练过程。

    参数:
    - channels: 张量的通道数。每个通道将独立进行归一化。
    - eps: 归一化过程中为了防止除以零添加的极小值。
    """

    def __init__(self, channels, eps=1e-6):
        """
        初始化二维层归一化模块。

        初始化权重和偏置参数，以及归一化过程中使用的epsilon值。
        """
        super(LayerNorm2d, self).__init__()
        # 初始化权重参数为1，用于归一化后的缩放。
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        # 初始化偏置参数为0，用于归一化后的偏移。
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        # 设置epsilon值，用于防止除以零。
        self.eps = eps

    def forward(self, x):
        """
        执行二维层归一化的前向传播。

        参数:
        - x: 输入的二维张量。

        返回:
        - 经过层归一化处理后的张量。
        """
        # 调用LayerNormFunction.apply执行具体的层归一化计算。
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)

class CABlock(nn.Module):
    """
    内容自适应块（Content Adaptive Block），用于根据输入内容进行自适应调整。

    该块主要通过自适应平均池化和1x1卷积来提取全局内容特征，并将其与原始输入特征图相乘，以实现内容自适应的特征增强。

    参数:
    - channels: 输入通道数，用于指定卷积层的输入和输出通道数。
    """

    def __init__(self, channels):
        """
        初始化内容自适应块。

        初始化一个自适应平均池化层和一个1x1卷积层，用于提取全局特征并进行通道间调整。
        """
        super(CABlock, self).__init__()
        # 初始化自适应平均池化层和1x1卷积层，用于提取全局特征
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1)
        )

    def forward(self, x):
        """
        前向传播函数。

        输入特征图x，通过自适应平均池化和1x1卷积提取全局特征，然后与原始输入特征图相乘，
        实现内容自适应的特征增强。

        参数:
        - x: 输入特征图。

        返回:
        - 输出特征图，经过内容自适应增强后的结果。
        """
        # 应用自适应平均池化和卷积操作，提取全局特征并进行通道间调整
        return x * self.ca(x)

class DualStreamGate(nn.Module):
    """
    双流门控类，用于将两个输入流进行组合。

    该类的作用是通过对输入的两个流x和y进行分割，然后将分割后的两个流进行乘法操作，
    实现流之间的交互。具体来说，它将x的第一个分量与y的第二个分量相乘，同时将x的第二个分量
    与y的第一个分量相乘，从而得到两个新的流。
    """

    def forward(self, x, y):
        # 将输入流x在维度1上分割为两个部分
        x1, x2 = x.chunk(2, dim=1)
        # 将输入流y在维度1上分割为两个部分
        y1, y2 = y.chunk(2, dim=1)
        # 返回两个流的乘积组合
        return x1 * y2, y1 * x2

class DualStreamSeq(nn.Sequential):
    """
    双流序列类，用于管理两个输入流的处理序列。

    该类继承自nn.Sequential，但在前向传播过程中处理两个输入流x和y。
    如果只有单个输入流x，它会将其复制为y。
    在处理每个模块时，两个流都会被传递给模块，并且每个模块应该能够处理两个输入流。
    """

    def forward(self, x, y=None):
        # 如果y为None，则将其设置为x，实现单流到双流的兼容
        y = y if y is not None else x
        # 遍历序列中的每个模块，同时处理两个输入流
        for module in self:
            x, y = module(x, y)
        # 返回处理后的两个流
        return x, y

class DualStreamBlock(nn.Module):
    """
    双流块类，用于封装一个包含多个模块的双流处理单元。

    该类的作用是封装一组模块，这些模块将按照顺序应用于两个输入流x和y。
    它可以接受一个OrderedDict来指定模块的顺序和名称，或者直接接受多个模块作为参数。
    """

    def __init__(self, *args):
        # 调用父类构造函数
        super(DualStreamBlock, self).__init__()
        # 初始化一个顺序容器来存储模块
        self.seq = nn.Sequential()
        # 根据参数类型来添加模块到顺序容器中
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            # 如果参数是一个OrderedDict，则按指定的顺序和名称添加模块
            for key, module in args[0].items():
                self.seq.add_module(key, module)
        else:
            # 如果参数是多个模块，则按顺序添加模块
            for idx, module in enumerate(args):
                self.seq.add_module(str(idx), module)

    def forward(self, x, y):
        # 应用顺序容器中的所有模块到两个输入流x和y，并返回处理后的结果
        return self.seq(x), self.seq(y)

class MuGIBlock(nn.Module):
    """
    MuGI块类，用于实现多路径信息交互的神经网络模块。

    参数:
    - c: 通道数。
    - shared_b: 是否共享偏置参数的布尔值。
    """

    def __init__(self, c, shared_b=False):
        super().__init__()
        # 初始化第一个双流序列，包含两个卷积块和一个注意力门机制。
        self.block1 = DualStreamSeq(
            DualStreamBlock(
                LayerNorm2d(c),
                nn.Conv2d(c, c * 2, 1),
                nn.Conv2d(c * 2, c * 2, 3, padding=1, groups=c * 2)
            ),
            DualStreamGate(),
            DualStreamBlock(CABlock(c)),
            DualStreamBlock(nn.Conv2d(c, c, 1))
        )

        # 初始化左侧和右侧的信息融合参数。
        self.a_l = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.a_r = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        # 初始化第二个双流序列，包含一个卷积块和一个注意力门机制。
        self.block2 = DualStreamSeq(
            DualStreamBlock(
                LayerNorm2d(c),
                nn.Conv2d(c, c * 2, 1)
            ),
            DualStreamGate(),
            DualStreamBlock(
                nn.Conv2d(c, c, 1)
            )

        )

        # 标记是否共享偏置参数。
        self.shared_b = shared_b
        # 根据是否共享偏置参数，初始化相应的偏置参数。
        if shared_b:
            self.b = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        else:
            self.b_l = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
            self.b_r = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp_l, inp_r):
        """
        前向传播函数，处理左右两个输入信号。

        参数:
        - inp_l: 左侧输入信号。
        - inp_r: 右侧输入信号。

        返回:
        - out_l: 左侧输出信号。
        - out_r: 右侧输出信号。
        """
        # 通过第一个双流序列处理输入。
        x, y = self.block1(inp_l, inp_r)
        # 使用融合参数融合信息，并保留跳过连接。
        x_skip, y_skip = inp_l + x * self.a_l, inp_r + y * self.a_r
        # 通过第二个双流序列进一步处理信号。
        x, y = self.block2(x_skip, y_skip)
        # 根据是否共享偏置参数，进行最后的融合。
        if self.shared_b:
            out_l, out_r = x_skip + x * self.b, y_skip + y * self.b
        else:
            out_l, out_r = x_skip + x * self.b_l, y_skip + y * self.b_r
        return out_l, out_r

# BaseModel——>SingleNetBaseModel——>LitModel（csecnet）
class tanh_L1Loss(nn.Module):
    def __init__(self):
        super(tanh_L1Loss, self).__init__()

    def forward(self, x, y):
        loss = torch.mean(torch.abs(torch.tanh(x) - torch.tanh(y)))
        return loss

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
                nn.Conv2d(in_features, in_features, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_features, in_features, 3, 1, 1)
        )

    def forward(self, x):
        return x + self.block(x)

    # 定义了DeepWBNet类，它继承自nn.Module


class SCAM(nn.Module):
    """通道注意力机制。

    参数：
        num_feat (int): 中间特征的通道数。
        squeeze_factor (int): 通道压缩因子。默认值：16。
    """

    def __init__(self, num_feat, squeeze_factor=3):  # 初始化函数
        super(SCAM, self).__init__()
        self.attention = nn.Sequential(  # 定义顺序容器
            nn.AdaptiveAvgPool2d(1),  # 自适应平均池化层
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),  # 卷积层降低通道数
            nn.ReLU(inplace=True),  # ReLU 激活函数
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),  # 卷积层恢复通道数
            nn.Sigmoid())  # Sigmoid 激活函数

    def forward(self, x):  # 前向传播函数
        y = self.attention(x)  # 计算注意力
        return x * y  # 将原始输入与注意力相乘以增强特征

# RCUNet空间一致性损失  RECnet 曝光对比正则化约束 颜色损失
class LitModel(SingleNetBaseModel):
    def __init__(self, opt):
        super().__init__(opt, DeepWBNet(opt[RUNTIME]), [TRAIN, VALID])
        # 初始化不同的损失函数，用于模型训练时的优化
        self.pixel_loss = tanh_L1Loss()# 像素级别的L1损失，经过tanh激活
        self.weighted_loss = WeightedL1Loss()# 加权的L1损失
        self.tvloss = L_TV()# 总变差损失，用于图像平滑
        self.ltv2 = LTVloss() # 另一种总变差损失
        self.cos = torch.nn.CosineSimilarity(1, 1e-8)
        # self.color = L_color() # 颜色损失
        # self.spa = L_spa() #实现空间损失
        self.histloss = HistogramLoss() # 直方图损失
        self.vggloss = VGGLoss(model="vgg16", shift=2)# VGG损失，使用VGG16模型
        self.vggloss.train()# 将VGG损失模型设置为训练模式
        self.inter_histloss = IntermediateHistogramLoss() # 中间直方图损失
        self.sparse_weight_loss = SparseWeightLoss()# 稀疏权重损失


    # 定义训练步骤的方法
    def training_step(self, batch, batch_idx):
        # 通过父类的训练步骤前向传播方法获取输入、真实值和输出
        input_batch, gt_batch, output_batch = super().training_step_forward(
            batch, batch_idx
        )
        # 定义一个字典，用于存储不同类型的损失函数及其计算方法
        loss_lambda_map = {
            # L1损失
            L1_LOSS: lambda: self.pixel_loss(output_batch, gt_batch),
            # 余弦相似度损失，通过1减去余弦相似度的平均值来计算
            # COLOR_LOSS: lambda: self.color(output_batch),
            # SPATIAL_LOSS: lambda: self.spa(output_batch, gt_batch),
            COS_LOSS: lambda: (1 - self.cos(output_batch, gt_batch).mean())
            * 0.5,
            COS_LOSS
            + "2": lambda: 1
            - F.sigmoid(self.cos(output_batch, gt_batch).mean()),
            # 总变差损失
            LTV_LOSS: lambda: self.tvloss(output_batch),
            # 针对特定中间结果的两种总变差损失
            "tvloss1": lambda: self.tvloss(self.net.res[ILLU_MAP])
            + self.tvloss(self.net.res[BRIGHTEN_INPUT]),
            "tvloss2": lambda: self.tvloss(self.net.res[INVERSE_ILLU_MAP])
            + self.tvloss(self.net.res[DARKEN_INPUT]),
            # 使用新的总变差损失函数计算两种损失
            "tvloss1_new": lambda: self.ltv2(
                input_batch, self.net.res[ILLU_MAP], 1
            ),
            "tvloss2_new": lambda: self.ltv2(
                1 - input_batch, self.net.res[INVERSE_ILLU_MAP], 1
            ),
            # 照明图损失，计算照明图与其逆的差的均方误差
            "illumap_loss": lambda: F.mse_loss(
                self.net.res[ILLU_MAP], 1 - self.net.res[INVERSE_ILLU_MAP]
            ),
            # 加权L1损失
            WEIGHTED_LOSS: lambda: self.weighted_loss(
                input_batch.detach(), output_batch, gt_batch
            ),
            # 结构相似性损失
            SSIM_LOSS: lambda: kn.losses.ssim_loss(
                output_batch, gt_batch, window_size=11
            ),
            # 峰值信噪比损失
            PSNR_LOSS: lambda: kn.losses.psnr_loss(
                output_batch, gt_batch, max_val=1.0
            ),
            # 直方图损失
            HIST_LOSS: lambda: self.histloss(output_batch, gt_batch),
            # 中间直方图损失
            INTER_HIST_LOSS: lambda: self.inter_histloss(
                input_batch,
                gt_batch,
                self.net.res[BRIGHTEN_INPUT],
                self.net.res[DARKEN_INPUT],
            ),
            # VGG损失
            VGG_LOSS: lambda: self.vggloss(input_batch, gt_batch),
        }
        # COSE 如果配置中启用了变形（DEFORM），则添加额外的损失
        if self.opt[RUNTIME][DEFORM]:
            loss_lambda_map.update(
                {
                    NORMAL_EX_LOSS: lambda: self.pixel_loss(
                        self.net.res[NORMAL], gt_batch
                    )
                }
            )
            # 计算并记录损失
        loss = self.calc_and_log_losses(loss_lambda_map)

        # logging images:# 记录训练过程中的图像
        self.log_training_iogt_img(batch)
        return loss

    def validation_step(self, batch, batch_idx): ...


    def test_step(self, batch, batch_ix):
        # 调用父类的测试步骤方法
        super().test_step(batch, batch_ix)

        # 保存中间结果
        for k, v in self.net.res.items():
            dirpath = Path(self.opt[IMG_DIRPATH]) / k  # 构建保存路径
            fname = os.path.basename(batch[INPUT_FPATH][0])  # 获取文件名
            util.mkdir(dirpath)  # 确保目录存在

            if "illu" in k:  # 如果键名包含"illu"，表示是照明图
                torchvision.utils.save_image(v[0].unsqueeze(1), dirpath / fname)  # 保存照明图像
            elif k == "guide_features":  # 如果是引导特征
                max_size = v[-1][-1].shape[-2:]  # 获取最大尺寸
                final = []  # 用于存储最终特征图
                for level_guide in v:  # 遍历每个层级的引导特征
                    gs = [F.interpolate(g, max_size) for g in level_guide]  # 上采样到最大尺寸
                    final.extend(gs)  # 添加到最终列表中
                region_num = final[0].shape[1]  # 获取区域数量
                final = torch.stack(final).argmax(axis=2).float() / region_num  # 计算最终特征图
                torchvision.utils.save_image(final, dirpath / fname)  # 保存图像
            elif k in {ILLU_MAP, INVERSE_ILLU_MAP, BRIGHTEN_INPUT, DARKEN_INPUT}:  # 如果是其他重要结果
                # 直接保存处理后的图像
                torchvision.utils.save_image(v, dirpath / fname)  # 保存图像
            elif k == NORMAL:

                torchvision.utils.save_image(v, dirpath / fname)  # 保存图像
            else:
                self.save_img_batch(v, dirpath, fname)  # 保存图像批次




class DeepWBNet(nn.Module):
    # 定义一个方法用于构建照明网络（illumination network）
    def build_illu_net(self):
        # 从当前目录下的bilateralupsamplenet模块中导入BilateralUpsampleNet类。
        from .bilateralupsamplenet import BilateralUpsampleNet
        # 使用从配置字典self.opt中获取的BUNET键对应的值作为参数，实例化BilateralUpsampleNet
        return BilateralUpsampleNet(self.opt[BUNET])


    # 定义一个方法用于执行骨干网络的前向传播
    def backbone_forward(self, net, x):
        # 对输入x进行下采样
        low_x = self.down_sampler(x)
        # 将下采样后的low_x和原始x输入到网络net中，得到结果res
        res = net(low_x, x)
        # 更新self.res字典，存储网络net的guide_features
        self.res.update({"guide_features": net.guide_features})
        # 检查和调试信息
        # print(f"low_x shape: {low_x.shape}, low_x min: {low_x.min()}, low_x max: {low_x.max()}")
        # print(f"res shape: {res.shape}, res min: {res.min()}, res max: {res.max()}")
        return res

    # 类的初始化方法
    def __init__(self, opt=None,
                 input_resolution=256,
                 sampler_input_resolution=256,
                 sampler_output_resolution=256,
                 n_vertices_3d=17,
                 n_vertices_2d=0,
                 n_vertices_1d=0
                 ):
        super(DeepWBNet, self).__init__()
        # 存储配置信息
        self.opt = opt
        # 初始化一个空字典，用于存储中间结果
        self.res = {}
        # 定义一个下采样器，使用双三次插值将输入x的大小调整为256x256

        self.down_sampler = Down_wt(3,3)
        # 自适应采样
        # self.down_sampler = MySampler(
        #     sampler_input_resolution=sampler_input_resolution, sampler_output_resolution=sampler_output_resolution)  # 采样器
        # 实例化照明网络
        self.illu_net = self.build_illu_net()
        # 定义输出网络，使用非局部块提高效率
        # Use non-local block for efficiency.
        nf = 32  # 特征通道数
        self.out_net = nn.Sequential(
            nn.Conv2d(9, nf, 3, 1, 1),# 输入通道数为9，输出通道数为nf，卷积核大小为3
            # 激活函数，ReLU（Rectified Linear Unit），inplace=True表示直接在原变量上进行修改，节省内存
            nn.ReLU(inplace=True),
            # 第二个卷积层，输入和输出通道数都为nf，卷积核大小为3x3，步长为1，填充为1
            # 同样，输出尺寸将与输入相同
            nn.Conv2d(nf, nf, 3, 1, 1),
            # 另一个ReLU激活函数
            nn.ReLU(inplace=True),
            # 非局部块，用于捕捉长距离依赖关系。输入通道数为nf，使用双线性插值进行下采样（如果需要的话），不使用批归一化层
            NONLocalBlock2D(nf, sub_sample="bilinear", bn_layer=False), # 非局部块
            # 1x1卷积层，输入通道数为nf，输出通道数仍为nf（但可以通过权重调整特征）
            # 1x1卷积常用于改变通道数或进行特征融合，而不改变空间尺寸
            nn.Conv2d(nf, nf, 1),  # 1x1卷积
            # 又一个ReLU激活函数
            nn.ReLU(inplace=True),
            # 最后一个卷积层，将通道数从nf转换为3（通常用于将特征映射回RGB图像空间）
            # 卷积核大小为1x1，因此不会改变空间尺寸
            nn.Conv2d(nf, 3, 1),  # 输出通道数为3
            # 另一个非局部块，但这次是在输出通道数为3的特征图上应用的
            # 注意：在实际应用中，将非局部块放在最后可能不是最佳选择，因为它可能会在最终的输出上引入不必要的复杂性
            # 这取决于具体的任务和模型设计
            NONLocalBlock2D(3, sub_sample="bilinear", bn_layer=False),  # 另一个非局部块
        )
        self.normal_out_net = nn.Sequential(
            nn.Conv2d(9, nf, 3, 1, 1),
            nn.BatchNorm2d(nf),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(nf, nf, 3, 1, 1),
            nn.BatchNorm2d(nf),
            nn.LeakyReLU(inplace=True),
            ResidualBlock(nf),
            ResidualBlock(nf),
            nn.Conv2d(nf, nf, 3, 1, 1),
            nn.BatchNorm2d(nf),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(nf, 3, 1),
            nn.Sigmoid()
        )

        # If you have enough computing resources, you can also use the following code to replace the above code.
        # Remember to modify the corresponding code in the forward function.
        # self.out_net = CrossNonLocalBlock(in_channels=3, inter_channels=16)

        # COSE  如果配置中启用了可变形卷积（DEFORM），则实例化两个ColorDeformConv2d层
        if opt[DEFORM]:
            self.over_deform = ColorDeformConv2d(
                inc=3,
                outc=3,
                kernel_size=3,
                padding=1,
                stride=1,
                modulation=True,
                color_deform=True,
            )
            self.under_deform = ColorDeformConv2d(
                inc=3,
                outc=3,
                kernel_size=3,
                padding=1,
                stride=1,
                modulation=True,
                color_deform=True,
            )

            self.mugi = MuGIBlock(3)
            self.scam = SCAM(3)  # 实例化 WAEM 模块
            self.cit = MEC_CIT()
            self.ODConv2d = ODConv2d()
            # self.hwd = Down_wt()
    # 定义一个方法用于分解输入x1和照明图illu_map
    # def decomp(self, x1, illu_map):
    #     # 计算分解后的结果，避免除以0的情况
    #     # return x1 / (torch.where(illu_map < x1, x1, illu_map.float()) + 1e-7)
    #     # 改进的分解方法，加入平滑处理，避免除以0和粗暴分解
    #     smooth_illu = torch.nn.functional.interpolate(illu_map, scale_factor=0.5, mode='bilinear', align_corners=False)
    #     smooth_illu = torch.nn.functional.interpolate(smooth_illu, scale_factor=2, mode='bilinear', align_corners=False)
    #     return x1 / (torch.where(smooth_illu < x1, x1, smooth_illu.float()) + 1e-7)

    def decomp(self, x1, illu_map):
        smooth_illu = F.avg_pool2d(illu_map, kernel_size=3, stride=1, padding=1)
        smooth_illu = TF.gaussian_blur(smooth_illu, kernel_size=5, sigma=2)
        # 确保 smooth_illu 在合理范围内
        smooth_illu = torch.clamp(smooth_illu, 0.1, 0.9)
        return x1 / (torch.where(smooth_illu < x1, x1, smooth_illu.float()) + 1e-7)

    # 定义前向传播方法
    def forward(self, x):
        # 假设 x 是一个 PyTorch 张量，代表一张图片
        # print(x.shape)

        # 原始输入x1和它的补集inverse_x1
        # x = self.scam(x)
        # 假设 x 是一个 PyTorch 张量，代表一张图片
        # print(x.shape)

        x1 = x

        inverse_x1 = 1 - x1

        # Backbone:
        # ──────────────────────────────────────────────────────────
        # 通过照明网络处理x1和inverse_x1，得到照明图和逆照明图
        illu_map = self.backbone_forward(self.illu_net, x1)
        inverse_illu_map = self.backbone_forward(self.illu_net, inverse_x1)
        print(f"illu_map grad: {illu_map.grad}")
        print(f"inverse_illu_map grad: {inverse_illu_map.grad}")
        print(f" illu_map min={illu_map.min()}, max={illu_map.max()}")
        print(f" inverse_illu_map min={inverse_illu_map.min()}, max={inverse_illu_map.max()}")

        # Enhancement:
        # ──────────────────────────────────────────────────────────
        # 将RGB照明图和逆照明图转换为灰度图
        illu_map = util.rgb2gray(illu_map)
        inverse_illu_map = util.rgb2gray(inverse_illu_map)
        # 使用分解方法处理x1和照明图，得到增强后的图像brighten_x1
        brighten_x1 = self.decomp(x1, illu_map)
        # 使用分解方法处理inverse_x1和逆照明图，得到增强后的图像（再取反得到darken_x1）
        inverse_x2 = self.decomp(inverse_x1, inverse_illu_map)
        darken_x1 = 1 - inverse_x2
        brighten_x1_od = self.ODConv2d(brighten_x1)
        darken_x1_od = self.ODConv2d(darken_x1)
        print(f" brighten_x1_od min={brighten_x1_od.min()}, max={brighten_x1_od.max()}")
        print(f" darken_x1_od min={darken_x1_od.min()}, max={darken_x1_od.max()}")
        brighten_x1=self.cit(brighten_x1)# 先进行一次CTB处理
        darken_x1=self.cit(darken_x1)# 先进行一次CTB处理
        # 更新self.res字典，存储中间结果
        self.res.update(
            {
                INVERSE: 1 - x,
                ILLU_MAP: illu_map,
                INVERSE_ILLU_MAP: inverse_illu_map,
                BRIGHTEN_INPUT: brighten_x1,
                DARKEN_INPUT: darken_x1,
            }
        )


        # ──────────────────────────────────────────────────────────
        # 将原始输入x、增强后的brighten_x1和darken_x1拼接后输入到输出网络，得到权重图weight_map

        weight_map = self.normal_out_net(torch.cat([x, brighten_x1, darken_x1], dim=1))

        # 从weight_map中分离出三个通道的权重w1, w2, w3
        w1 = weight_map[:, 0, ...].unsqueeze(1)
        w2 = weight_map[:, 1, ...].unsqueeze(1)
        w3 = weight_map[:, 2, ...].unsqueeze(1)

        # 根据权重图计算输出图像out
        out = x * w1 + brighten_x1 * w2 + darken_x1 * w3
        pseudo_normal=out

        self.res.update({NORMAL: pseudo_normal})
        pseudo_normal_od = self.ODConv2d(pseudo_normal)
        print(f" pseudo_normal_od min={pseudo_normal_od.min()}, max={pseudo_normal_od.max()}")
        # Deformation:变形部分
        # ──────────────────────────────────────────────────────────
        # 使用可变形卷积得到变形后的图像brighten_x2和darken_x2
        brighten_x2 = self.over_deform(x=pseudo_normal_od, ref=brighten_x1_od)
        darken_x2 = self.under_deform(x=pseudo_normal_od, ref=darken_x1_od)
        brighten_x2 = self.cit(brighten_x2)
        darken_x2 = self.cit(darken_x2)
        # brighten_x2, darken_x2 = self.mugi(brighten_x2, darken_x2)
        # 更新self.res字典，存储变形后的图像

        # ──────────────────────────────────────────────────────────
        # pseudo_normal、变形后的图像brighten_x2和darken_x2拼接后输入到输出网络，得到新的权重图
        x = self.scam(x)
        out = self.out_net(
            torch.cat([x, brighten_x2, darken_x2], dim=1)
        )
        # 再次从新的权重图中分离出三个通道的权重w1, w2, w3
        w1 = out[:, 0, ...].unsqueeze(1)
        w2 = out[:, 1, ...].unsqueeze(1)
        w3 = out[:, 2, ...].unsqueeze(1)

        # 根据新的权重图重新计算输出图像out
        out = x * w1 + brighten_x2 * w2 + darken_x2 * w3


        # out = self.lut(x, out)  # 获取LUT输出

        # 更新self.res字典，存储变形后的图像
        self.res.update(
            {
                BRIGHTEN_OFFSET: brighten_x2,
                DARKEN_OFFSET: darken_x2,
            }
        )

        # 确保输出图像out的形状与输入图像x的形状相同
        assert out.shape == x.shape,f"Output shape {out.shape} does not match input shape {x.shape}"
        return out