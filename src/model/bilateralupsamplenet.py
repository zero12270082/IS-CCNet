import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(project_root))
# from model import parse_model_class
sys.path.insert(0, '../')
from src.globalenv import *
from .arch.drconv import DRConv2d
from src.model.arch.unet_based.hist_unet import HistUNet
from .basic_loss import LTVloss
from .single_net_basemodel import SingleNetBaseModel



# 定义LitModel类，继承自SingleNetBaseModel
class LitModel(SingleNetBaseModel):
    # 初始化方法，接收配置信息
    def __init__(self, opt):
        # 调用父类的初始化方法，传入指定参数
        super().__init__(opt, BilateralUpsampleNet(opt[RUNTIME]), [TRAIN, VALID])
        low_res = opt[RUNTIME][LOW_RESOLUTION]  # 从选项中获取低分辨率值

        # 定义下采样函数，使用双三次插值将输入x的大小调整为低分辨率
        self.down_sampler = lambda x: F.interpolate(
            x, size=(low_res, low_res), mode="bicubic", align_corners=False
        )
        self.use_illu = opt[RUNTIME][PREDICT_ILLUMINATION]  # 获取是否使用光照预测的标志

        # 定义损失函数
        self.mse = torch.nn.MSELoss()  # 均方误差损失
        self.ltv = LTVloss()  # 自定义LTV损失
        self.cos = torch.nn.CosineSimilarity(1, 1e-8)  # 余弦相似度损失

        self.net.train()  # 设置模型为训练模式

    # 训练步骤
    def training_step(self, batch, batch_idx):
        # 执行父类的前向传播步骤
        input_batch, gt_batch, output_batch = super().training_step_forward(batch, batch_idx)
        # 定义损失计算的映射字典
        loss_lambda_map = {
            MSE: lambda: self.mse(output_batch, gt_batch),  # 均方误差
            COLOR_LOSS: lambda: self.color(output_batch),  # 颜色损失
            SPATIAL_LOSS: lambda: self.spa(output_batch, gt_batch),  # 空间损失
            LTV_LOSS: lambda: (
                self.ltv(input_batch, self.net.illu_map, 1)  # LTV损失（如果使用光照预测）
                if self.use_illu
                else None
            ),
        }

        # 计算损失并记录日志
        loss = self.calc_and_log_losses(loss_lambda_map)
        self.log_training_iogt_img(
            batch,
            extra_img_dict={
                PREDICT_ILLUMINATION: self.net.illu_map,  # 记录预测的光照图
                GUIDEMAP: self.net.guidemap,  # 记录引导图
            },
        )
        return loss  # 返回计算的损失

    # 验证步骤
    def validation_step(self, batch, batch_idx):
        # 执行父类的验证步骤
        super().validation_step(batch, batch_idx)

    # 测试步骤
    def test_step(self, batch, batch_ix):
        # 执行父类的测试步骤
        super().test_step(batch, batch_ix)

    # 前向传播方法
    def forward(self, x):
        low_res_x = self.down_sampler(x)  # 使用下采样器将输入调整为低分辨率
        return self.net(low_res_x, x)  # 将低分辨率和原始输入传入网络


# 定义卷积块类
class ConvBlock(nn.Module):
    # 初始化方法，定义卷积块的参数
    def __init__(
        self,
        inc,
        outc,
        kernel_size=3,
        padding=1,
        stride=1,
        use_bias=True,
        activation=nn.ReLU,
        batch_norm=False,
    ):
        super(ConvBlock, self).__init__()
        conv_type = OPT["conv_type"]  # 获取卷积类型
        # 判断卷积类型并初始化对应的卷积层
        if conv_type == "conv":
            self.conv = nn.Conv2d(
                int(inc),
                int(outc),
                kernel_size,
                padding=padding,
                stride=stride,
                bias=use_bias,
            )
        elif conv_type.startswith("drconv"):
            region_num = int(conv_type.replace("drconv", ""))  # 获取区域数
            self.conv = DRConv2d(
                int(inc),
                int(outc),
                kernel_size,
                region_num=region_num,
                padding=padding,
                stride=stride,
            )
        else:
            raise NotImplementedError()  # 未实现的卷积类型抛出异常

        self.activation = activation() if activation else None  # 初始化激活函数
        self.bn = nn.BatchNorm2d(outc) if batch_norm else None  # 初始化批归一化层

    # 前向传播方法
    def forward(self, x):
        x = self.conv(x)  # 执行卷积操作
        if self.bn:  # 如果有批归一化，则应用它
            x = self.bn(x)
        if self.activation:  # 如果有激活函数，则应用它
            x = self.activation(x)
        return x  # 返回处理后的结果


# 定义SliceNode类
class SliceNode(nn.Module):
    # 初始化方法，接收配置信息
    def __init__(self, opt):
        super(SliceNode, self).__init__()
        self.opt = opt  # 存储配置信息

    # 前向传播方法，接收双边网格和引导图
    def forward(self, bilateral_grid, guidemap):
        # bilateral_grid形状: Nx12x8x16x16
        device = bilateral_grid.get_device()  # 获取设备信息
        N, _, H, W = guidemap.shape  # 从引导图获取批次大小和高宽
        hg, wg = torch.meshgrid([torch.arange(0, H), torch.arange(0, W)])  # 创建网格 [0, H-1] 和 [0, W-1]
        if device >= 0:  # 如果在 GPU 上执行
            hg = hg.to(device)  # 将hg转移到相应设备
            wg = wg.to(device)  # 将wg转移到相应设备

        hg = (
            hg.float().repeat(N, 1, 1).unsqueeze(3) / (H - 1) * 2 - 1
        )  # 归一化到[-1,1]的范围
        wg = (
            wg.float().repeat(N, 1, 1).unsqueeze(3) / (W - 1) * 2 - 1
        )  # 归一化到[-1,1]的范围
        guidemap = guidemap * 2 - 1  # 归一化引导图
        guidemap = guidemap.permute(0, 2, 3, 1).contiguous()  # 调整引导图的维度顺序
        guidemap_guide = torch.cat([wg, hg, guidemap], dim=3).unsqueeze(1)  # 拼接并添加维度

        guidemap_guide = guidemap_guide.type_as(bilateral_grid)  # 确保类型与双边网格一致
        coeff = F.grid_sample(
            bilateral_grid, guidemap_guide, "bilinear", align_corners=True
        )  # 使用双线性插值从双边网格中采样得到系数
        return coeff.squeeze(2)  # 返回去掉指定维度的系数


# 定义ApplyCoeffs类
class ApplyCoeffs(nn.Module):
    def __init__(self):
        super(ApplyCoeffs, self).__init__()

    # 前向传播方法，接收系数和原始输入
    def forward(self, coeff, full_res_input):
        """
        coeff形状: [bs, 12, h, w]
        input形状: [bs, 3, h, w]
            Affine:
            r = a11*r + a12*g + a13*b + a14
            g = a21*r + a22*g + a23*b + a24
            ...
        """
        R = (
            torch.sum(full_res_input * coeff[:, 0:3, :, :], dim=1, keepdim=True)  # 计算红色通道
            + coeff[:, 3:4, :, :]  # 加上偏置
        )
        G = (
            torch.sum(full_res_input * coeff[:, 4:7, :, :], dim=1, keepdim=True)  # 计算绿色通道
            + coeff[:, 7:8, :, :]  # 加上偏置
        )
        B = (
            torch.sum(
                full_res_input * coeff[:, 8:11, :, :], dim=1, keepdim=True  # 计算蓝色通道
            )
            + coeff[:, 11:12, :, :]  # 加上偏置
        )

        return torch.cat([R, G, B], dim=1)  # 合并三个通道并返回


# 定义GuideNet类
class GuideNet(nn.Module):
    def __init__(self, params=None, out_channel=1):
        super(GuideNet, self).__init__()
        self.params = params  # 存储参数
        self.conv1 = ConvBlock(3, 16, kernel_size=1, padding=0, batch_norm=True)  # 第一层卷积块
        self.conv2 = ConvBlock(16, out_channel, kernel_size=1, padding=0, activation=nn.Sigmoid)  # 第二层卷积块

    # 前向传播方法
    def forward(self, x):
        return self.conv2(self.conv1(x))  # 依次通过两层卷积块


# 定义LowResHistUNet类
class LowResHistUNet(HistUNet):
    # 初始化方法，接收系数维度和配置信息
    def __init__(self, coeff_dim=12, opt=None):
        super(LowResHistUNet, self).__init__(
            in_channels=3,
            out_channels=coeff_dim * opt[LUMA_BINS],
            bilinear=True,
            **opt[HIST_UNET],
        )
        self.coeff_dim = coeff_dim  # 存储系数维度

    # 前向传播方法
    def forward(self, x):
        y = super(LowResHistUNet, self).forward(x)  # 调用父类的前向传播
        y = torch.stack(torch.split(y, self.coeff_dim, 1), 2)  # 将输出按系数维度分割并堆叠
        return y  # 返回处理后的结果


# 定义BilateralUpsampleNet类
class BilateralUpsampleNet(nn.Module):
    def __init__(self, opt):
        super(BilateralUpsampleNet, self).__init__()
        self.opt = opt  # 存储配置信息
        global OPT
        OPT = opt  # 设置全局配置
        self.guide = GuideNet(params=opt)  # 实例化引导网络
        self.slice = SliceNode(opt)  # 实例化切片节点
        self.build_coeffs_network(opt)  # 建立系数网络

    # 建立系数网络
    def build_coeffs_network(self, opt):
        Backbone = LowResHistUNet  # 选择低分辨率直方图U-Net作为主干网络
        self.coeffs = Backbone(opt=opt)  # 实例化系数网络
        self.apply_coeffs = ApplyCoeffs()  # 实例化应用系数的模块

    # 前向传播方法
    def forward(self, lowres, fullres):
        bilateral_grid = self.coeffs(lowres)  # 从低分辨率输入获取双边网格
        try:
            self.guide_features = self.coeffs.guide_features  # 尝试获取引导特征
        except Exception as e:
            print("[ WARN ] {}".format(e))  # 捕获并打印警告信息
        guide = self.guide(fullres)  # 从全分辨率输入生成引导图
        self.guidemap = guide  # 存储引导图

        slice_coeffs = self.slice(bilateral_grid, guide)  # 从双边网格和引导图中切片获取系数
        out = self.apply_coeffs(slice_coeffs, fullres)  # 应用系数计算输出

        self.slice_coeffs = slice_coeffs  # 存储切片系数
        self.illu_map = None  # 初始化光照图为空

        return out  # 返回最终输出
