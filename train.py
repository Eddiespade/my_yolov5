import os  # 与操作系统进行交互的模块 包含文件路径操作和解析
import random  # 生成随机数模块
import numpy as np  # numpy数组操作模块
import torch  # Pytorch深度学习相关模块
import yaml  # 操作yaml文件模块
from pathlib import Path  # Path将str转换为Path对象 使字符串路径易于操作的模块

if __name__ == '__main__':
    # opt设置文件
    with open("configs/opt.yaml", encoding='utf-8') as f:
        opt = yaml.safe_load(f)  # load hyps dict  加载超参信息
    # 输出超参信息 opt: ...
    # 显示颜色 print("\033[显示方式；前景颜色；背景颜色m 要打印的信息 \033[0m")
    print('\033[0;31;40mopt:\033[0m ' + ', '.join(f'{k}={v} ' for k, v in opt.items()))

    save_dir = Path(opt['project'])
    print(save_dir)
    weights_path = 'weights' + os.sep
    last = weights_path + 'last.pt'
    best = weights_path + 'best.pt'
    results_file = weights_path + 'results.txt'

    # 初始化随机数种子, 以保证结果的一致性
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
