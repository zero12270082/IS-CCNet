# # -*- coding: utf-8 -*-
# import os
# import time
#
# import hydra
# import pytorch_lightning as pl
# from omegaconf import open_dict
# from pytorch_lightning import Trainer
# import sys
# project_root = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.dirname(project_root))
# # from model import parse_model_class
# sys.path.insert(0, '../')
# from globalenv import *
# from src.utils.util import parse_config
#
# pl.seed_everything(GLOBAL_SEED)
#
#
# @hydra.main(config_path="config", config_name="config")
# def main(opt):
#     opt = parse_config(opt, TEST)
#     print("Running config:", opt)
#     from model.csecnet import LitModel as ModelClass
#
#     ckpt = opt[CHECKPOINT_PATH]
#     assert ckpt
#     model = ModelClass.load_from_checkpoint(ckpt, opt=opt)
#     with open_dict(opt):
#         model.opt[IMG_DIRPATH] = model.build_test_res_dir()
#         opt.mode = "test"
#     print(f"Loading model from: {ckpt}")
#
#     from data.img_dataset import DataModule
#
#     datamodule = DataModule(opt)
#
#     trainer = Trainer(
#         accelerator="gpu",
#         devices=-1,
#     )
#     # trainer = Trainer(
#     #     gpus=opt[GPU], strategy=opt[BACKEND], precision=opt[RUNTIME_PRECISION]
#     # )
#     beg = time.time()
#     trainer.test(model, datamodule)
#     print(f"[ TIMER ] Total time usage: {time.time() - beg}")
#     print(f"[ PATH ] The results are saved in: {model.opt[IMG_DIRPATH]}")
#
#
# if __name__ == "__main__":
#     main()
# -*- coding: utf-8 -*-
import os
import time
import hydra
import pytorch_lightning as pl
from omegaconf import open_dict
from pytorch_lightning import Trainer
import sys

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(project_root))
sys.path.insert(0, '../')
from globalenv import *
from src.utils.util import parse_config

pl.seed_everything(GLOBAL_SEED)

@hydra.main(config_path="config", config_name="config")
def main(opt):
    # 解析配置并打印
    opt = parse_config(opt, TEST)
    print("Running config:", opt)

    # 加载模型
    from model.csecnet import LitModel as ModelClass
    ckpt = opt[CHECKPOINT_PATH]
    assert ckpt, "请提供有效的检查点路径！"
    model = ModelClass.load_from_checkpoint(ckpt, opt=opt)

    # 配置模型的测试模式和结果保存路径
    with open_dict(opt):
        model.opt[IMG_DIRPATH] = model.build_test_res_dir()
        opt.mode = "test"
    print(f"Loading model from: {ckpt}")

    # 初始化数据模块
    from data.img_dataset import DataModule
    datamodule = DataModule(opt)

    # 创建Trainer对象，配置为使用GPU
    trainer = Trainer(
        accelerator="gpu",
        devices=-1,
    )

    # 开始测试
    beg = time.time()
    trainer.test(model, datamodule)
    print(f"[ TIMER ] Total time usage: {time.time() - beg} seconds")
    print(f"[ PATH ] The results are saved in: {model.opt[IMG_DIRPATH]}")

if __name__ == "__main__":
    main()
