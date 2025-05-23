# -*- coding: utf-8 -*-
import os
import platform
import os
os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"

import hydra
import pytorch_lightning as pl
import torch
# 从omegaconf库导入open_dict函数，用于修改配置字典
from omegaconf import open_dict
# 从pytorch_lightning.callbacks导入回调函数，用于在训练过程中执行特定操作
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ModelSummary,
)
from pytorch_lightning.loggers import TensorBoardLogger
import sys
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(project_root))
# from model import parse_model_class
sys.path.insert(0, '../')
from data.img_dataset import DataModule
from globalenv import *
from src.utils.util import init_logging, parse_config
# 使用pytorch_lightning的seed_everything函数设置全局随机种子，确保实验的可重复性
pl.seed_everything(GLOBAL_SEED)

# 使用 Hydra 加载配置
@hydra.main(config_path="config", config_name="config")
def main(config):
    # 尝试打印GPU状态信息。
    try:
        print("GPU status info:")
        os.system("nvidia-smi")
    except:
        ...
    # 清空缓存
    torch.cuda.empty_cache()
    # 解析配置，并根据调试模式（DEBUG）调整配置
    opt = parse_config(config, TRAIN)
    if opt.name == DEBUG:
        opt.debug = True
    # 初始化TensorBoard日志记录器
    if opt.debug:
        mylogger = None
        # 如果提供了检查点路径，则加载检查点并获取继续训练的epoch
        if opt.checkpoint_path:
            continue_epoch = torch.load(
                opt.checkpoint_path, map_location=torch.device("cpu")
            )["global_step"]
            # 设置调试模式下的配置
        debug_config = {
            DATALOADER_N: 0,
            NAME: DEBUG,
            LOG_EVERY: 1,
            VALID_EVERY: 1,
            NUM_EPOCH: 2 if not opt.checkpoint_path else continue_epoch + 2,
        }
        # 更新配置
        opt.update(debug_config)
        # 打印调试模式警告和配置
        debug_str = "[red]>>>> [[ WARN ]] You are in debug mode, update configs. <<<<[/red]"
        print(f"{debug_str}\n{debug_config}\n{debug_str}")

    else:
        # 如果不是调试模式
        # rename the exp
        # 根据操作系统类型设置分隔符
        spl = "_" if platform.system() == "Windows" else ":"
        # 重命名实验
        opt.name = f"{opt.runtime.modelname}{spl}{opt.name}@{opt.train_ds.name}"

        # trainer logger. init early to record all console output.初始化TensorBoard日志记录器
        mylogger = TensorBoardLogger(
            name=opt.name,
            save_dir=ROOT_PATH / "tb_logs",
        )
    # 使用open_dict上下文管理器修改配置，并初始化日志目录和图片目录
    with open_dict(opt):
        opt.log_dirpath, opt.img_dirpath = init_logging(TRAIN, opt)

    # init logging# 打印当前运行的配置
    print("Running config:", opt)

    # load data# 初始化DataModule实例，用于加载数据
    datamodule = DataModule(opt)

    # callbacks:# 初始化回调函数列表，首先添加ModelSummary回调
    callbacks = [ModelSummary(max_depth=0)]
    if opt[EARLY_STOP]:
        print(# 如果配置中启用了EarlyStopping，则打印相关信息
            f"Apply EarlyStopping when `{opt.checkpoint_monitor}` is {opt.monitor_mode}"
        )
        callbacks.append(# 添加EarlyStopping回调
            EarlyStopping(opt.checkpoint_monitor, mode=opt.monitor_mode)
        )

    # callbacks: # 初始化ModelCheckpoint回调，用于保存最佳模型
    checkpoint_callback = ModelCheckpoint(
        dirpath=opt[LOG_DIRPATH],
        filename="{epoch}",
        save_last=True,
        save_top_k=5,
        mode=opt.monitor_mode,
        monitor=opt.checkpoint_monitor,
        save_on_train_epoch_end=True,
        every_n_epochs=opt.savemodel_every,
    )
    # 将ModelCheckpoint回调添加到回调列表中
    callbacks.append(checkpoint_callback)
    # 如果AMP后端不是native，则打印警告信息
    if opt[AMP_BACKEND] != "native":
        print(
            f"WARN: Running in APEX, mode: {opt[AMP_BACKEND]}-{opt[AMP_LEVEL]}"
        )
    else:# 否则，将AMP级别设置为None
        opt[AMP_LEVEL] = None
    # amp_backend = opt[AMP_BACKEND],
    # amp_level = opt[AMP_LEVEL],
    # strategy = opt[BACKEND],
    # init trainer:
    # 初始化Trainer对象，配置训练参数
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=-1,
        use_distributed_sampler=True,
        max_epochs=opt[NUM_EPOCH],
        logger=mylogger,
        callbacks=callbacks,
        check_val_every_n_epoch=opt[VALID_EVERY],
        num_sanity_val_steps=opt[VAL_DEBUG_STEP_NUMS],
        precision=opt[RUNTIME_PRECISION],
        **opt.flags,
    )
    print("Trainer initailized.")

    # training loop# 从model.csecnet模块导入LitModel类，作为模型类
    from model.csecnet import LitModel as ModelClass
    # 根据是否存在检查点路径和是否恢复训练，加载模型或继续训练
    # 如果提供了检查点路径且未启用继续训练，则加载检查点并从第0步开始训练。
    if opt.checkpoint_path and not opt.resume_training:
        print("Load ckpt and train from step 0...")
        model = ModelClass.load_from_checkpoint(opt.checkpoint_path, opt=opt)
        trainer.fit(model, datamodule)
        # print('------------------',model)

    else:# 否则，初始化模型实例，并根据是否提供了检查点路径来决定是否继续训练
        model = ModelClass(opt)
        print(f"Continue training: {opt.checkpoint_path}")
        trainer.fit(model, datamodule, ckpt_path=opt.checkpoint_path)
        # print('-------------------------------',model)


if __name__ == "__main__":
    main()
