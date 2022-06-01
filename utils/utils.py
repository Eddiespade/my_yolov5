import platform
from pathlib import Path  # Path将str转换为Path对象 使字符串路径易于操作的模块
import re           # 用来匹配字符串（动态、模糊）的模块
import glob         # 仅支持部分通配符的文件搜索模块
import os
import torch


def increment_path(path, sep='', mkdir=False):
    """这是个用处特别广泛的函数 train.py、detect.py、test.py等都会用到
    递增路径 如 run/train/exp --> runs/train/exp{sep}0, runs/exp{sep}1 etc.
    :params path: window path   run/train/exp
    :params exist_ok: False
    :params sep: exp文件名的后缀  默认''
    :params mkdir: 是否在这里创建dir  False
    """
    path = Path(path)  # string/win路径 -> win路径
    # 如果该文件夹已经存在 则将路径run/train/exp修改为 runs/train/exp1
    if path.exists():
        # path.suffix 得到路径path的后缀  ''
        suffix = path.suffix
        # .with_suffix 将路径添加一个后缀 ''
        path = path.with_suffix('')
        # 模糊搜索和path\sep相似的路径, 存在一个list列表中 如['runs\\train\\exp', 'runs\\train\\exp1']
        # f开头表示在字符串内支持大括号内的python表达式
        dirs = glob.glob(f"{path}{sep}*")
        # r的作用是去除转义字符       path.stem: 没有后缀的文件名 exp
        # re 模糊查询模块  re.search: 查找dir中有字符串'exp/数字'的d   \d匹配数字
        # matches [None, <re.Match object; span=(11, 15), match='exp1'>]  可以看到返回span(匹配的位置) match(匹配的对象)
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        # i = [1]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        # 生成需要生成文件的exp后面的数字 n = max(i) + 1 = 2
        n = max(i) + 1 if i else 1  # increment number
        # 返回path runs/train/exp2
        path = Path(f"{path}{sep}{n}{suffix}")  # update path

    # path.suffix文件后缀   path.parent　路径的上级目录  runs/train/exp2
    dir = path if path.suffix == '' else path.parent  # directory
    if not dir.exists() and mkdir:  # mkdir 默认False 先不创建dir
        dir.mkdir(parents=True, exist_ok=True)  # make directory
    return path  # 返回runs/train/exp2


def select_device(device='', batch_size=None):
    """广泛用于train.py、val.py、detect.py等文件中
    用于选择模型训练的设备 并输出日志信息
    :params device: 输入的设备  device = 'cpu' or '0' or '0,1,2,3'
    :params batch_size: 一个批次的图片个数
    """
    # git_describe(): 返回当前文件父文件的描述信息(yolov5)   date_modified(): 返回当前文件的修改日期
    # s: 之后要加入logger日志的显示信息
    s = f'\033[0;31;40mmy_yolov5: \033[0m' + f' torch {torch.__version__}  '  # string

    # 如果device输入为cpu  cpu=True  device.lower(): 将device字符串全部转为小写字母
    cpu = device.lower() == 'cpu'
    if cpu:
        # 如果cpu=True 就强制(force)使用cpu 令torch.cuda.is_available() = False
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    elif device:
        # 如果输入device不为空  device=GPU  直接设置 CUDA environment variable = device 加入CUDA可用设备
        os.environ['CUDA_VISIBLE_DEVICES'] = device
        # 检查cuda的可用性 如果不可用则终止程序
        assert torch.cuda.is_available(), f'CUDA unavailable, invalid device {device} requested'

    # 输入device为空 自行根据计算机情况选择相应设备  先看GPU 没有就CPU
    # 如果cuda可用 且 输入device != cpu 则 cuda=True 反正cuda=False
    cuda = not cpu and torch.cuda.is_available()
    if cuda:
        # devices: 如果cuda可用 返回所有可用的gpu设备 i.e. 0,1,6,7  如果不可用就返回 '0'
        devices = device.split(',') if device else '0'
        # n: 所有可用的gpu设备数量  device count
        n = len(devices)
        # 检查是否有gpu设备 且 batch_size是否可以能被显卡数目整除  check batch_size is divisible by device_count
        if n > 1 and batch_size:
            # 如果不能则关闭程序
            assert batch_size % n == 0, f'batch-size {batch_size} not multiple of GPU count {n}'

        space = ' ' * (len(s) + 1)  # 定义等长的空格

        # 满足所有条件 s加上所有显卡的信息
        for i, d in enumerate(devices):
            # p: 每个可用显卡的相关属性
            p = torch.cuda.get_device_properties(i)
            # 显示信息s加上每张显卡的属性信息
            s += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / 1024 ** 2}MB)\n"  # bytes to MB
    else:
        # cuda不可用显示信息s就加上CPU
        s += 'CPU\n'

    print(s)
    # 如果cuda可用就返回第一张显卡的的名称 如: GeForce RTX 2060 反之返回CPU对应的名称
    return torch.device('cuda:0' if cuda else 'cpu')