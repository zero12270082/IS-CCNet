import torch


def CE(rgb_img, increment):
    # 对传入的 RGB 图像进行复制，方便后续修改
    img = rgb_img * 1.0

    # 计算图像在每个像素的通道（R, G, B）上的最小值和最大值
    img_min = torch.amin(img, dim=2)  # 直接返回最小值
    img_max = torch.amax(img, dim=2)  # 直接返回最大值
    img_out = img.clone()  # 初始化输出图像为输入图像的副本

    # 获取图像的饱和度和亮度在 HSL 颜色空间中的计算
    delta = (img_max - img_min) / 255.0  # 计算颜色通道间的差异归一化到 [0, 1]
    value = (img_max + img_min) / 255.0  # 计算亮度值
    L = value / 2.0  # 计算亮度的中间值

    # 根据亮度选择饱和度的计算方式
    mask_1 = L < 0.5  # 布尔掩码：检查像素亮度是否小于 0.5
    s1 = delta / (value)  # 计算饱和度的第一种方法
    s2 = delta / (2 - value)  # 计算饱和度的第二种方法
    s = s1 * mask_1 + s2 * (1 - mask_1.float())  # 将 mask_1 转换为浮点类型

    # 检查增量是否大于或等于 0，从而增强饱和度
    if increment >= 0:
        # 计算临时饱和度值并创建布尔掩码以选择的 alpha 值
        temp = increment + s  # 增加饱和度的增量
        mask_2 = temp > 1  # 布尔掩码：检查临时饱和度是否超过 1
        alpha_1 = s  # 当临时饱和度超过 1 时的 alpha 值
        alpha_2 = s * 0 + 1 - increment  # 当临时饱和度未超过 1 时的 alpha 值
        alpha = alpha_1 * mask_2 + alpha_2 * (1 - mask_2.float())  # 将 mask_2 转换为浮点类型

        # 通过倒数进行饱和度的增强
        alpha = 1 / alpha - 1  # 将 alpha 转化为增益值
        # 调整输出图像的各个颜色通道
        img_out = img_out.clone()  # 创建副本
        img_out[:, :, 0] = img[:, :, 0] + (img[:, :, 0] - L * 255.0) * alpha  # 调整红色通道
        img_out[:, :, 1] = img[:, :, 1] + (img[:, :, 1] - L * 255.0) * alpha  # 调整绿色通道
        img_out[:, :, 2] = img[:, :, 2] + (img[:, :, 2] - L * 255.0) * alpha  # 调整蓝色通道

    # 如果增量小于 0，饱和度线性衰减
    else:
        alpha = increment  # 使用给定的负增量作为 alpha
        # 调整输出图像的各个颜色通道
        img_out = img_out.clone()  # 创建副本
        img_out[:, :, 0] = img[:, :, 0] + (img[:, :, 0] - L * 255.0) * alpha  # 调整红色通道
        img_out[:, :, 1] = img[:, :, 1] + (img[:, :, 1] - L * 255.0) * alpha  # 调整绿色通道
        img_out[:, :, 2] = img[:, :, 2] + (img[:, :, 2] - L * 255.0) * alpha  # 调整蓝色通道

    img_out = img_out / 255.0  # 将输出图像值归一化到 [0, 1]

    # RGB 颜色的上下限处理（小于 0 的取 0，大于 1 的取 1）
    mask_3 = img_out < 0  # 小于 0 的布尔掩码
    mask_4 = img_out > 1  # 大于 1 的布尔掩码
    img_out = img_out * (1 - mask_3.float())  # 将 mask_3 转换为浮点类型
    img_out = img_out * (1 - mask_4.float()) + mask_4.float()  # 将 mask_4 转换为浮点类型

    return img_out  # 返回处理后的图像