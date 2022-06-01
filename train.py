import os  # 与操作系统进行交互的模块 包含文件路径操作和解析
import random  # 生成随机数模块
import numpy as np  # numpy数组操作模块
import torch  # Pytorch深度学习相关模块
import yaml  # 操作yaml文件模块
from pathlib import Path  # Path将str转换为Path对象 使字符串路径易于操作的模块

from models.yolo import Model
from utils.utils import increment_path, select_device

if __name__ == '__main__':
    # 初始化随机数种子, 以保证结果的一致性
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    """ 以下为加载 configs下的 配置文件"""
    with open("configs/opt.yaml", encoding='utf-8') as f:   # 读取opt设置文件 ---> 对应 yolov5中的 argparse
        opt = yaml.safe_load(f)  # 加载设置信息
    # 输出超参信息 opt: ... 显示颜色 print("\033[显示方式；前景颜色；背景颜色m 要打印的信息 \033[0m")
    print('\033[0;31;40mopt:\033[0m ' + ', '.join(f'{k}={v} ' for k, v in opt.items()))

    with open(opt['hyp'], encoding='utf-8') as f:           # Hyperparameters超参
        hyp = yaml.safe_load(f)  # 加载超参信息
    print('\033[0;31;40mHyperparameters:\033[0m ' + ', '.join(f'{k}={v} ' for k, v in opt.items()))

    # data_dict: 加载VOC.yaml中的数据配置信息  dict
    with open(opt['data'], encoding='utf-8') as f:
        data_dict = yaml.safe_load(f)

    # 自增保存路径
    save_dir = Path(str(increment_path(Path(opt['project']) / opt['name'])))  # 保存训练结果的目录  如runs/train/exp18
    weights_dir = save_dir / 'weights'  # 保存权重路径 如runs/train/exp18/weights
    # 创建文件目录   参数说明：parents：如果父目录不存在，是否创建父目录。 exist_ok：只有在目录不存在时创建目录，目录已存在时不会抛出异常。
    weights_dir.mkdir(parents=True, exist_ok=True)
    last = weights_dir / 'last.pt'
    best = weights_dir / 'best.pt'
    results_file = save_dir / 'results.txt'

    # 搭建模型
    device = select_device(opt['device'], batch_size=opt['batch_size'])
    nc = 1 if opt['single_cls'] else int(data_dict['nc'])  # number of classes
    model = Model(opt['cfg'], ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create
