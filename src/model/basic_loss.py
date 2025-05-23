# 导入需要的库
import kornia as kn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

# 从自定义模块中导入计算直方图的函数
from .arch.hist import get_hist, get_hist_conv

# 重新导入torch和功能库（此行可以省略，因为前面已经导入过）
import torch
import torch.nn.functional as F

# 定义重建损失函数，包括光照和反射分量
def reconstruction_loss(output_illumination, output_reflection, input_image):
    # 假设 output_illumination 和 output_reflection 是通过某种方式从网络输出得到的
    # 这里使用L1损失来引导网络产生清晰的光照和反射分量
    return F.l1_loss(output_illumination, input_image.max(dim=1, keepdim=True)[0]) + \
           F.l1_loss(output_reflection, input_image / output_illumination)

# 定义空间一致性损失函数
def spatial_consistency_loss(input_image, enhanced_image, patch_size=4):
    # 将图像划分为 patch_size x patch_size 的小块，并计算相邻块之间的差异
    # 这里简化处理，只考虑水平和垂直方向的差异
    def compute_patch_difference(image, dx, dy):
        # 对输入图像进行填充，创建移动图像
        padded_image = F.pad(image, (dy, -dy, dx, -dx))
        shifted_image = padded_image[..., dy:image.size(-1) + dy, dx:image.size(-2) + dx]
        # 计算原始块和移动后的块之间的L1损失
        return F.l1_loss(image, shifted_image, reduction='sum')

    # 获取输入图像的高和宽
    h, w = input_image.size()[-2:]
    total_loss = 0
    # 对整个图像进行块处理
    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            # 提取增强后的图像的一块
            patch = enhanced_image[..., i:i + patch_size, j:j + patch_size]
            # 计算当前块的空间一致性损失
            total_loss += compute_patch_difference(patch, 0, patch_size) + compute_patch_difference(patch, patch_size, 0)
    # 将损失归一化到图像大小
    return total_loss / (h * w)

# 定义颜色恒定损失函数
def color_constancy_loss(enhanced_image):
    # 计算增强图像各通道的平均值，并使他们接近同一灰度值
    means = enhanced_image.mean(dim=(2, 3), keepdim=True)
    gray_value = means.mean(dim=1, keepdim=True)
    # 计算均方误差损失
    return F.mse_loss(means, gray_value)

# 定义噪声损失函数
def noise_loss(enhanced_image):
    # 通过计算图像梯度，近似噪声损失（使用拉普拉斯算子）
    laplacian = torch.nn.functional.conv2d(enhanced_image,
                                           torch.tensor([[[[0, 1, 0], [1, -4, 1], [0, 1, 0]]]], dtype=torch.float32),
                                           padding=1)
    # 计算梯度的L2范数作为噪声损失
    return laplacian.pow(2).mean()

# 定义曝光损失函数
def exposure_loss(enhanced_image, target_exposure=0.5):
    # 根据目标曝光值调整图像的亮度，并计算L1损失
    target_mean = target_exposure * torch.ones_like(enhanced_image.mean(dim=(2, 3), keepdim=True))
    # 计算均值的L1损失
    return F.l1_loss(enhanced_image.mean(dim=(2, 3), keepdim=True), target_mean)

class L_spa(nn.Module):
    """实现空间损失的类。

    参数：
        loss_weight (float): 损失的权重。默认值为 1.0。
    """

    def __init__(self, loss_weight=1.0):
        super(L_spa, self).__init__()  # 初始化 nn.Module
        self.loss_weight = loss_weight  # 保存损失权重

        # 定义局部方向上的卷积核
        kernel_left = torch.FloatTensor([[0, 0, 0], [-1, 1, 0], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_right = torch.FloatTensor([[0, 0, 0], [0, 1, -1], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_up = torch.FloatTensor([[0, -1, 0], [0, 1, 0], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_down = torch.FloatTensor([[0, 0, 0], [0, 1, 0], [0, -1, 0]]).cuda().unsqueeze(0).unsqueeze(0)

        # 将卷积核设置为不可训练的参数
        self.weight_left = nn.Parameter(data=kernel_left, requires_grad=False)
        self.weight_right = nn.Parameter(data=kernel_right, requires_grad=False)
        self.weight_up = nn.Parameter(data=kernel_up, requires_grad=False)
        self.weight_down = nn.Parameter(data=kernel_down, requires_grad=False)
        self.pool = nn.AvgPool2d(4)  # 定义平均池化层

    def forward(self, org, enhance):
        """前向传播函数。

        参数：
            org: 原始图像
            enhance: 增强图像

        返回：
            E: 计算得到的损失
        """
        b, c, h, w = org.shape  # 获取输入特征的形状

        org_mean = torch.mean(org, 1, keepdim=True)  # 计算原始图像的均值
        enhance_mean = torch.mean(enhance, 1, keepdim=True)  # 计算增强图像的均值

        org_pool = self.pool(org_mean)  # 对原始均值进行池化
        enhance_pool = self.pool(enhance_mean)  # 对增强均值进行池化

        # 计算权重差
        weight_diff = torch.max(
            torch.FloatTensor([1]).cuda() + 10000 * torch.min(org_pool - torch.FloatTensor([0.3]).cuda(),
                                                              torch.FloatTensor([0]).cuda()),
            torch.FloatTensor([0.5]).cuda()
        )

        # 计算增强图像与池化原始图像的差异
        E_1 = torch.mul(torch.sign(enhance_pool - torch.FloatTensor([0.5]).cuda()), enhance_pool - org_pool)

        # 计算原始图像的方向导数
        D_org_letf = F.conv2d(org_pool, self.weight_left, padding=1)
        D_org_right = F.conv2d(org_pool, self.weight_right, padding=1)
        D_org_up = F.conv2d(org_pool, self.weight_up, padding=1)
        D_org_down = F.conv2d(org_pool, self.weight_down, padding=1)

        # 计算增强图像的方向导数
        D_enhance_letf = F.conv2d(enhance_pool, self.weight_left, padding=1)
        D_enhance_right = F.conv2d(enhance_pool, self.weight_right, padding=1)
        D_enhance_up = F.conv2d(enhance_pool, self.weight_up, padding=1)
        D_enhance_down = F.conv2d(enhance_pool, self.weight_down, padding=1)

        # 计算方向导数的差异的平方
        D_left = torch.pow(D_org_letf - D_enhance_letf, 2)
        D_right = torch.pow(D_org_right - D_enhance_right, 2)
        D_up = torch.pow(D_org_up - D_enhance_up, 2)
        D_down = torch.pow(D_org_down - D_enhance_down, 2)

        # 求和得到最终的损失
        E = (D_left + D_right + D_up + D_down)

        return self.loss_weight * E  # 返回加权后的损失


class L_color(nn.Module):
    """实现颜色损失的类。

    参数：
        loss_weight (float): 损失的权重。默认值为 1.0。
    """

    def __init__(self, loss_weight=1.0):
        super(L_color, self).__init__()  # 初始化 nn.Module
        self.loss_weight = loss_weight  # 保存损失权重

    def forward(self, x):
      b, c, h, w = x.shape  # 获取输入特征的形状
  
      mean_rgb = torch.mean(x, [2, 3], keepdim=True)  # 计算每个通道的平均值
      mr, mg, mb = torch.split(mean_rgb, 1, dim=1)  # 分离 RGB 通道的均值
      Drg = torch.pow(mr - mg, 2)  # 计算红绿通道的差异的平方
      Drb = torch.pow(mr - mb, 2)  # 计算红蓝通道的差异的平方
      Dgb = torch.pow(mb - mg, 2)  # 计算蓝绿通道的差异的平方
  
      # 计算最终的颜色损失
      # k = torch.pow(torch.pow(Drg, 2) + torch.pow(Drb, 2) + torch.pow(Dgb, 2), 0.5)
      k = torch.sqrt(Drg + Drb + Dgb).mean()
      # 将颜色损失的形状改变为 [batch_size, 1, 128, 128]
      # k = k.view(b, 1, 1, 1)  # 保持这一点，确保我们有 batch_size
      # k = k.expand(b, 1, h, w)  # 将其扩展为目标形状
      # # 调整 k 的形状为 [1, 1, 128, 128]
      # k = F.interpolate(k, size=(128, 128), mode='bilinear', align_corners=False)
    
    
      return self.loss_weight * k  # 返回加权后的颜色损失


# 定义直方图损失类
class HistogramLoss(nn.Module):
    # 初始化方法，设置直方图的箱数和下采样比例
    def __init__(self, n_bins=8, downscale=16):
        super().__init__()
        self.n_bins = n_bins  # 直方图的箱数
        self.hist_conv = get_hist_conv(n_bins, downscale)  # 获取处理直方图的卷积层

        # 定义将直方图张量打包的函数，将维度合并以适应后续计算
        self.pack_tensor = lambda x: x.reshape(
            self.n_bins, -1, *x.shape[-2:]
        ).permute(1, 0, 2, 3)

    # 前向传播方法，计算损失
    def forward(self, output, gt):
        gt_hist = get_hist(gt, self.n_bins)  # 计算目标直方图
        output_hist = get_hist(output, self.n_bins)  # 计算输出直方图

        shrink_hist_gt = self.hist_conv(self.pack_tensor(gt_hist))  # 下采样目标直方图
        shrink_hist_output = self.hist_conv(self.pack_tensor(output_hist))  # 下采样输出直方图

        # 计算直方图的均方误差损失
        return F.mse_loss(shrink_hist_gt, shrink_hist_output)

# 定义中间直方图损失类，继承自HistogramLoss
class IntermediateHistogramLoss(HistogramLoss):
    # 初始化方法
    def __init__(self, n_bins=8, downscale=16):
        super().__init__(n_bins, downscale)
        self.exposure_threshold = 0.5  # 曝光阈值

    # 前向传播方法
    def forward(self, img, gt, brighten, darken):
        """
        输入增强的亮部和暗部图像，获取两者之间的损失：
        - 亮部图像与目标的暗部区域
        - 暗部图像与目标的亮部区域
        """
        bs, c, _, _ = gt.shape  # 获取批次大小和通道数
        gt_hist = get_hist(gt, self.n_bins)  # 计算目标直方图
        shrink_hist_gt = self.hist_conv(self.pack_tensor(gt_hist))  # 下采样目标直方图

        down_size = shrink_hist_gt.shape[-2:]  # 获取下采样后的尺寸
        shrink_hist_gt = shrink_hist_gt.reshape(bs, c, self.n_bins, *down_size)  # 重塑目标直方图
        down_x = F.interpolate(img, size=down_size)  # 下采样输入图像

        # 获取曝光掩码
        over_ixs = down_x > self.exposure_threshold  # 亮部索引
        under_ixs = down_x <= self.exposure_threshold  # 暗部索引
        over_mask = down_x.clone()  # 亮部掩码
        over_mask[under_ixs] = 0
        over_mask[over_ixs] = 1
        over_mask.unsqueeze_(2)  # 增加维度
        under_mask = down_x.clone()  # 暗部掩码
        under_mask[under_ixs] = 1
        under_mask[over_ixs] = 0
        under_mask.unsqueeze_(2)  # 增加维度

        # 下采样暗部和亮部图像的直方图
        shrink_darken_hist = self.hist_conv(
            self.pack_tensor(get_hist(darken, self.n_bins))
        ).reshape(bs, c, self.n_bins, *down_size)
        shrink_brighten_hist = self.hist_conv(
            self.pack_tensor(get_hist(brighten, self.n_bins))
        ).reshape(bs, c, self.n_bins, *down_size)

        # 使用SSIM损失计算
        return 0.5 * kn.losses.ssim_loss(
            (shrink_hist_gt * over_mask).view(-1, c, *down_size),  # 亮部掩码下的目标直方图
            (shrink_darken_hist * over_mask).view(-1, c, *down_size),
            window_size=5,
        ) + 0.5 * kn.losses.ssim_loss(
            (shrink_hist_gt * under_mask).view(-1, c, *down_size),  # 暗部掩码下的目标直方图
            (shrink_brighten_hist * under_mask).view(-1, c, *down_size),
            window_size=5,
        )

# 定义加权 L1 损失类
class WeightedL1Loss(nn.Module):
    def __init__(self):
        super().__init__()

    # 前向传播方法
    def forward(self, input, output, gt):
        bias = 0.1  # 微小偏差
        # 计算权重
        weights = (torch.abs(input - 0.5) + bias) / 0.5
        weights = weights.mean(axis=1).unsqueeze(1).repeat(1, 3, 1, 1)  # 扩展维度以适应输出
        # 计算加权损失
        loss = torch.mean(torch.abs(output - gt) * weights.detach())
        return loss

# 定义LTV损失类
class LTVloss(nn.Module):
    def __init__(self, alpha=1.2, beta=1.5, eps=1e-4):
        super(LTVloss, self).__init__()
        self.alpha = alpha  # 参数 alpha
        self.beta = beta    # 参数 beta
        self.eps = eps      # 防止除零的微小值

    # 前向传播方法
    def forward(self, origin, illumination, weight):
        """
        origin:       一批输入数据, shape [batchsize, 3, h, w]
        illumination: 一批预测的光照数据
        """

        # 计算亮度I
        I = (
            origin[:, 0:1, :, :] * 0.299
            + origin[:, 1:2, :, :] * 0.587
            + origin[:, 2:3, :, :] * 0.114
        )
        L = torch.log(I + self.eps)  # 计算亮度的对数

        # 计算x方向和y方向的梯度
        dx = L[:, :, :-1, :-1] - L[:, :, :-1, 1:]
        dy = L[:, :, :-1, :-1] - L[:, :, 1:, :-1]

        # 计算梯度的归一化权重
        dx = self.beta / (torch.pow(torch.abs(dx), self.alpha) + self.eps)
        dy = self.beta / (torch.pow(torch.abs(dy), self.alpha) + self.eps)

        # 计算L2损失
        x_loss = dx * (
            (illumination[:, :, :-1, :-1] - illumination[:, :, :-1, 1:]) ** 2
        )
        y_loss = dy * (
            (illumination[:, :, :-1, :-1] - illumination[:, :, 1:, :-1]) ** 2
        )
        # 计算总体损失
        tvloss = torch.mean(x_loss + y_loss) / 2.0

        return tvloss * weight  # 根据权重返回损失

# 定义L_TV损失类
class L_TV(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(L_TV, self).__init__()
        self.TVLoss_weight = TVLoss_weight  # 设置损失权重

    # 前向传播方法
    def forward(self, x):
        batch_size = x.size()[0]  # 获取批次大小
        h_x = x.size()[2]         # 获取图像高度
        w_x = x.size()[3]         # 获取图像宽度
        count_h = (x.size()[2] - 1) * x.size()[3]  # 计算高度的数量
        count_w = x.size()[2] * (x.size()[3] - 1)  # 计算宽度的数量
        # 计算高度方向的总变差
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, : h_x - 1, :]), 2).sum()
        # 计算宽度方向的总变差
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, : w_x - 1]), 2).sum()
        return (
            self.TVLoss_weight
            * 2
            * (h_tv / count_h + w_tv / count_w)
            / batch_size  # 将损失归一化到批次
        )

# 定义稀疏权重损失类
class SparseWeightLoss(nn.Module):
    def __init__(self, sparse_weight_loss_weight=1):
        super().__init__()
        self.sparse_weight_loss_weight = sparse_weight_loss_weight  # 设置损失权重

    # 前向传播方法
    def forward(self, weights):
        return self.sparse_weight_loss_weight * torch.mean(weights.pow(2))  # 计算权重的平方损失

# 定义VGG损失类
class VGGLoss(nn.Module):
    """计算输入图像与目标图像之间的VGG感知损失。

    输入和目标必须是四维张量，形状为``(B, 3, H, W)``，并且必须具有相同的形状。
    像素值应规范化到0到1的范围。

    VGG感知损失是输入和目标在指定层的特征之间的均方差（默认值为8，即``relu2_2``）。
    """

    models = {"vgg16": models.vgg16, "vgg19": models.vgg19}  # 可用的VGG模型

    # 初始化方法
    def __init__(self, model="vgg16", layer=8, shift=0, reduction="mean"):
        super().__init__()
        self.shift = shift  # 随机偏移
        self.reduction = reduction  # 损失计算方式
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]  # 图像归一化
        )
        self.model = self.models[model](pretrained=True).features[: layer + 1]  # 加载VGG模型
        self.model.eval()  # 设置为评估模式
        self.model.requires_grad_(False)  # 不需要计算梯度

    # 获取特征的方法
    def get_features(self, input):
        return self.model(self.normalize(input))  # 计算输入的VGG特征

    # 设置训练模式
    def train(self, mode=True):
        self.training = mode

    # 前向传播方法
    def forward(self, input, target, target_is_features=False):
        if target_is_features:
            input_feats = self.get_features(input)  # 计算输入特征
            target_feats = target  # 目标特征
        else:
            sep = input.shape[0]  # 获取批次大小
            batch = torch.cat([input, target])  # 合并输入和目标
            if self.shift and self.training:  # 如果启用偏移
                padded = F.pad(batch, [self.shift] * 4, mode="replicate")  # 填充
                batch = transforms.RandomCrop(batch.shape[2:])(padded)  # 随机裁剪
            feats = self.get_features(batch)  # 计算合并后的特征
            input_feats, target_feats = feats[:sep], feats[sep:]  # 分离输入和目标特征
        return F.mse_loss(input_feats, target_feats, reduction=self.reduction)  # 计算均方误差损失
