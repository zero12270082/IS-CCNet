import torch
import torch.nn as nn
from pytorch_wavelets import DWTForward  # 导入小波变换模块

class Down_wt(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Down_wt, self).__init__()  # 初始化父类 nn.Module
        # 初始化小波变换模块，J=1 表示一层小波分解，mode='zero' 表示边界填充方式，wave='haar' 表示使用 Haar 小波
        self.wt = DWTForward(J=1, mode='zero', wave='haar')
        # 定义卷积、批归一化和 ReLU 激活的序列模块
        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(in_ch*4, out_ch, kernel_size=1, stride=1),  # 1x1 卷积，输入通道数为 in_ch*4，输出通道数为 out_ch
            nn.BatchNorm2d(out_ch),  # 批归一化
            nn.ReLU(inplace=True),  # ReLU 激活函数，inplace=True 表示原地操作
        )

    def forward(self, x):
        # 对输入 x 进行一层小波变换，返回低频分量 yL 和高频分量 yH
        yL, yH = self.wt(x)
        # 提取高频分量的三个方向分量：水平 (y_HL)、垂直 (y_LH) 和对角 (y_HH)
        y_HL = yH[0][:,:,0,::]  # 水平方向高频分量
        y_LH = yH[0][:,:,1,::]  # 垂直方向高频分量
        y_HH = yH[0][:,:,2,::]  # 对角方向高频分量
        # 将低频分量 yL 和三个高频分量 y_HL, y_LH, y_HH 在通道维度上拼接
        x = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)
        # 将拼接后的特征通过卷积、批归一化和 ReLU 激活的序列模块
        x = self.conv_bn_relu(x)

        return x  # 返回处理后的特征