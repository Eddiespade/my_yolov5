import os  # 与操作系统进行交互的模块 包含文件路径操作和解析
import random  # 生成随机数模块
import numpy as np  # numpy数组操作模块
import torch  # Pytorch深度学习相关模块
import yaml  # 操作yaml文件模块
from pathlib import Path  # Path将str转换为Path对象 使字符串路径易于操作的模块

from utils.utils import increment_path

if __name__ == '__main__':
    # opt设置文件
    with open("configs/opt.yaml", encoding='utf-8') as f:
        opt = yaml.safe_load(f)  # load hyps dict  加载超参信息
    # 输出超参信息 opt: ...
    # 显示颜色 print("\033[显示方式；前景颜色；背景颜色m 要打印的信息 \033[0m")
    print('\033[0;31;40mopt:\033[0m ' + ', '.join(f'{k}={v} ' for k, v in opt.items()))

    # 自增保存路径
    save_dir = Path(str(increment_path(Path(opt['project']) / opt['name'])))
    weights_dir = save_dir / 'weights'
    last = weights_dir / 'last.pt'
    best = weights_dir / 'best.pt'
    results_file = save_dir / 'results.txt'

    # 初始化随机数种子, 以保证结果的一致性
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
