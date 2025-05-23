# -*- coding: utf-8 -*-
import os
import os.path as osp

import cv2
import torch
import torch.nn.functional as F
import torch.optim as optim
import sys
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(project_root))
import src.utils.util as util
from src.globalenv import *

from .basemodel import BaseModel
# BaseModel——>SingleNetBaseModel——>LitModel（csecnet）
# 定义一个名为SingleNetBaseModel的类，它继承自BaseModel。这个类用于只有一个self.net的网络模型
class SingleNetBaseModel(BaseModel):
    # for models with only one self.net
    # 初始化方法，接收优化选项opt、网络模型net、运行模式running_modes和是否打印网络架构print_arch
    def __init__(self, opt, net, running_modes, print_arch=False):
        # 调用父类的初始化方法
        super().__init__(opt, running_modes)
        # 将传入的网络模型赋值给self.net
        self.net = net
        # 将网络模型设置为训练模式
        self.net.train()

        # config for SingleNetBaseModel# 如果print_arch为True，则打印网络架构
        if print_arch:
            print(str(net))
            # 创建一个色调映射器，使用Reinhard色调映射算法，gamma值为2.2
        self.tonemapper = cv2.createTonemapReinhard(2.2)

        # training step forward# 初始化迭代计数器
        self.cnt_iters = 1

    # 配置优化器的方法
    def configure_optimizers(self):
        # self.parameters in LitModel is the same as nn.Module.
        # once you add nn.xxxx as a member in __init__, self.parameters will include it.
        # 使用Adam优化器，学习率从self.learning_rate获取（这个值应该在父类初始化时设置或从opt中获取）
        optimizer = optim.Adam(self.net.parameters(), lr=self.learning_rate,weight_decay=1e-5)
        # 使用余弦退火学习率调度器，T_max设置为10
        schedular = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=10
        )
        # 返回优化器和调度器的列表
        return [optimizer], [schedular]

    # 前向传播方法，接收输入x并返回网络的输出
    def forward(self, x):
        return self.net(x)

    # 训练步骤前向传播方法，接收一个批次的数据batch和批次索引batch_idx
    def training_step_forward(self, batch, batch_idx):
        # 清空CUDA缓存
        torch.cuda.empty_cache()
        # 如果模型尚未被监控，并且不在调试模式下，且日志记录器为wandb，则开始监控模型
        if (
            not self.MODEL_WATCHED
            and not self.opt[DEBUG]
            and self.opt.logger == "wandb"
        ):
            self.logger.experiment.watch(
                self.net, log_freq=self.opt[LOG_EVERY] * 2, log_graph=True
            )
            self.MODEL_WATCHED = True
        # 从批次数据中提取输入和真实标签
        input_batch, gt_batch = batch[INPUT], batch[GT]
        # 通过网络前向传播得到输出
        output_batch = self(input_batch)
        # 遍历网络返回的结果字典，根据键名提取特定的输出
        for k, v in self.net.res.items():
            if ILLU_MAP == k:
                illu_map = v
            elif INVERSE == k:
                inverse = v
            elif INVERSE_ILLU_MAP == k:
                inverse_illu_map = v
            elif BRIGHTEN_INPUT == k:
                brighten_input = v
            elif DARKEN_INPUT == k:
                darken_input = v
            elif NORMAL == k:
                normal_ex = v
            elif BRIGHTEN_OFFSET == k:
                brighten_offset = v
            elif DARKEN_OFFSET == k:
                darken_offset = v
            else:
                continue
                # 将输入、输出、真实标签和提取的特定输出组合成一个字典
        self.iogt = {
            INPUT: input_batch,
            ILLU_MAP: illu_map,
            BRIGHTEN_INPUT: brighten_input,
            BRIGHTEN_OFFSET: brighten_offset,
            NORMAL: normal_ex,
            INVERSE_ILLU_MAP: inverse_illu_map,
            DARKEN_INPUT: darken_input,
            DARKEN_OFFSET: darken_offset,
            OUTPUT: output_batch,
            GT: gt_batch,
        }
        # 返回输入、真实标签和网络输出，用于后续的计算和损失计算
        return input_batch, gt_batch, output_batch

    # 定义一个名为validation_step的方法，它接受两个参数：batch（批次数据）和batch_idx（批次索引）。
    # 这个方法的具体实现在这里被省略了（用...表示），它通常用于在验证阶段处理每个批次的数据。
    def validation_step(self, batch, batch_idx): ...

    # 定义一个名为on_validation_start的方法，它在验证过程开始时被调用
    # 这个方法初始化两个列表：total_psnr和total_ssim，用于存储验证过程中的PSNR（峰值信噪比）和SSIM（结构相似性）指标
    def on_validation_start(self):
        self.total_psnr = []
        self.total_ssim = []

    # 定义一个名为on_validation_end的方法，它在验证过程结束时被调用。
    # 这个方法的具体实现在这里被省略了（用...表示），它通常用于处理验证结束时的逻辑，如计算平均PSNR和SSIM
    def on_validation_end(self): ...

    # 定义一个名为log_training_iogt_img的方法，它接受两个参数：batch（批次数据）和extra_img_dict（额外的图像字典，默认为None）。
    # 这个方法仅在training_step中使用，用于记录训练过程中的图像信息。
    def log_training_iogt_img(self, batch, extra_img_dict=None):
        """
        Only used in training_step
        """
        # 如果提供了额外的图像字典，则将其与self.iogt合并，否则直接使用self.iogt
        if extra_img_dict:
            img_dict = {**self.iogt, **extra_img_dict}
        else:
            img_dict = self.iogt
        # 如果当前的全局步骤（global_step）是配置中指定的日志记录间隔（LOG_EVERY）的倍数，则记录图像
        if self.global_step % self.opt[LOG_EVERY] == 0:
            self.log_images_dict(
                mode=TRAIN,# 指定记录模式为训练模式
                input_fname=osp.basename(batch[INPUT_FPATH][0]),# 记录输入图像的文件名
                img_batch_dict=img_dict,# 要记录的图像字典
                gt_fname=osp.basename(batch[GT_FPATH][0]),# 记录真实（ground truth）图像的文件名
            )

    # 定义一个静态方法logdomain2hdr，它将低动态范围（LDR）图像批次转换为高动态范围（HDR）图像。
    # 这个方法接受一个参数ldr_batch，表示LDR图像批次。
    @staticmethod
    def logdomain2hdr(ldr_batch):
        # 使用公式10^ldr_batch - 1将LDR图像转换为HDR图像。
        # 这个转换是基于HDR图像通常以对数域的形式存储，转换回线性域时需要应用此公式。
        return 10**ldr_batch - 1

    # 定义一个名为on_test_start的方法，它在测试过程开始时被调用。
    # 这个方法初始化三个变量：total_psnr（用于存储每个测试批次的PSNR值），total_ssim（用于存储每个测试批次的SSIM值），以及global_test_step（用于跟踪测试的全局步骤）
    def on_test_start(self):
        self.total_psnr = []
        self.total_ssim = []
        self.global_test_step = 0

    # 定义一个名为on_test_end的方法，它在测试过程结束时被调用。
    # 这个方法打印出测试的总步数、平均PSNR值和平均SSIM值。
    def on_test_end(self):
        print(
            f"Test step: {len(self.total_psnr)}, Manual PSNR: {sum(self.total_psnr) / len(self.total_psnr)}, Manual SSIM: {sum(self.total_ssim) / len(self.total_ssim)}"
        )

    # 定义一个名为test_step的方法，它接受两个参数：batch（批次数据）和batch_ix（批次索引，但在方法体内未使用）。
    # 这个方法用于在测试阶段处理每个批次的数据，包括保存测试结果、计算PSNR和SSIM（当有真实图像GT时）
    def test_step(self, batch, batch_ix):
        """
        save test result and calculate PSNR and SSIM for `self.net` (when have GT)
        """
        # test without GT image:
        # 测试没有GT图像的情况：
        self.global_test_step += 1# 增加全局测试步骤计数器
        input_batch = batch[INPUT]# 从批次数据中获取输入图像
        assert input_batch.shape[0] == 1# 断言输入批次中只有一个图像（这可能是为了简化处理）
        output_batch = self(input_batch)# 通过模型处理输入图像，得到输出图像
        save_num = 1 # 设置要保存的图像数量（这里固定为1）

        # 测试有GT的情况：
        # test with GT:
        if GT in batch:# 如果批次数据中包含GT图像
            gt_batch = batch[GT]# 从批次数据中获取GT图像
            if output_batch.shape != batch[GT].shape: # 如果输出图像和GT图像的形状不匹配
                print(
                    f"[[ WARN ]] output.shape is {output_batch.shape} but GT.shape is {batch[GT].shape}. Resize GT to output to get PSNR."
                )
                gt_batch = F.interpolate(batch[GT], output_batch.shape[2:])# 使用插值方法调整GT图像的形状

            # 将CUDA张量转换为NumPy数组，以便进行图像处理和计算PSNR/SSIM
            output_ = util.cuda_tensor_to_ndarray(output_batch)
            y_ = util.cuda_tensor_to_ndarray(gt_batch)
            # 计算PSNR和SSIM
            psnr = util.ImageProcessing.compute_psnr(output_, y_, 1.0)
            ssim = util.ImageProcessing.compute_ssim(output_, y_)
            # 将计算得到的PSNR和SSIM添加到对应的列表中
            self.total_psnr.append(psnr)
            self.total_ssim.append(ssim)

        # save images# 保存图像
        self.save_img_batch(
            output_batch, # 要保存的输出图像批次
            self.opt[IMG_DIRPATH],  # 保存图像的目录路径（从配置中获取）
            osp.basename(batch[INPUT_FPATH][0]),  # 保存图像时使用的基础文件名（从输入图像路径中获取）
            save_num=save_num,  # 要保存的图像数量
        )
