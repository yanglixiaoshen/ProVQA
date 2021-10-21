import numpy as np
import os
import random
import time
import torch
from os import path as osp

from .dist_util import master_only
from .logger import get_root_logger


def set_random_seed(seed):
    """Set random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_time_str():
    return time.strftime('%Y%m%d_%H%M%S', time.localtime())


def mkdir_and_rename(path):
    """mkdirs. If path exists, rename it with timestamp and create a new one.

    Args:
        path (str): Folder path.
    """
    # /media/yl/yl_8t/bvqa360_experiments/experiments/0301_yl_train_BVQA360_Basic_ODV-VQA720
    if osp.exists(path): # 如果存在这个路径，就把它重命名；如果不存在，可能改配置后名字变了就直接新建一个目录。
        new_name = path + '_archived_' + get_time_str()
        print(f'Path already exists. Rename it to {new_name}', flush=True)
        os.rename(path, new_name)
    os.makedirs(path, exist_ok=True) # 最开始训练会在这一步新建'experiments_root'目录;或者修改将上一步的目录改时间戳后，新建当前训练的目录


@master_only
def make_exp_dirs(opt):
    """Make dirs for experiments."""
    path_opt = opt['path'].copy() # 列表，元组和字典的淺拷貝，意味著新列表中的任何修改都不會反映到原始列表中
    if opt['is_train']:  # 'experiments_root'：/media/yl/yl_8t/bvqa360_experiments/experiments/0301_yl_train_BVQA360_Basic_ODV-VQA720
        mkdir_and_rename(path_opt.pop('experiments_root')) # Python 字典 pop() 方法删除字典给定键 key 及对应的值，返回值为被删除的值。key 值必须给出。
    else:
        mkdir_and_rename(path_opt.pop('results_root'))
    for key, path in path_opt.items(): # 上一步将原来的experiments目录重命名作为旧的目录，并且新建了新的exp目录；这一步将models,visual.,trainingst等目录重新新建在当前exp目录下
        if ('strict_load' not in key) and ('pretrain_network' not in key) and ('resume'  not in key):
            # 除了 strict_load, pretrain_network, resume 这三个条目下不需要新建,剩下的models,visual,trainingst等目录重新新建在当前experiments_root目录下
            os.makedirs(path, exist_ok=True)  # 最开始训练会在这一步新建exp目录下级的models等目录




def scandir(dir_path, suffix=None, recursive=False, full_path=False):
    """Scan a directory to find the interested files.

    Args:
        dir_path (str): Path of the directory.
        suffix (str | tuple(str), optional): File suffix that we are
            interested in. Default: None.
        recursive (bool, optional): If set to True, recursively scan the
            directory. Default: False.
        full_path (bool, optional): If set to True, include the dir_path.
            Default: False.

    Returns:
        A generator for all the interested files with relative pathes.
    """

    if (suffix is not None) and not isinstance(suffix, (str, tuple)):
        raise TypeError('"suffix" must be a string or tuple of strings')

    root = dir_path

    def _scandir(dir_path, suffix, recursive):
        for entry in os.scandir(dir_path):
            if not entry.name.startswith('.') and entry.is_file():
                if full_path:
                    return_path = entry.path
                else:
                    return_path = osp.relpath(entry.path, root)

                if suffix is None:
                    yield return_path
                elif return_path.endswith(suffix):
                    yield return_path
            else:
                if recursive:
                    yield from _scandir(
                        entry.path, suffix=suffix, recursive=recursive)
                else:
                    continue

    return _scandir(dir_path, suffix=suffix, recursive=recursive)


def check_resume(opt, resume_iter):
    """Check resume states and pretrain_network paths.

    Args:
        opt (dict): Options.
        resume_iter (int): Resume iteration.
    """
    logger = get_root_logger()
    if opt['path']['resume_state']:
        # get all the networks
        networks = [key for key in opt.keys() if key.startswith('network_')]
        flag_pretrain = False
        for network in networks:
            if opt['path'].get(f'pretrain_{network}') is not None:
                flag_pretrain = True
        if flag_pretrain:
            logger.warning(
                'pretrain_network path will be ignored during resuming.')
        # set pretrained model paths
        for network in networks:
            name = f'pretrain_{network}'
            basename = network.replace('network_', '')
            if opt['path'].get('ignore_resume_networks') is None or (
                    basename not in opt['path']['ignore_resume_networks']):
                opt['path'][name] = osp.join(
                    opt['path']['models'], f'net_{basename}_{resume_iter}.pth')
                logger.info(f"Set {name} to {opt['path'][name]}")


def sizeof_fmt(size, suffix='B'):
    """Get human readable file size.

    Args:
        size (int): File size.
        suffix (str): Suffix. Default: 'B'.

    Return:
        str: Formated file siz.
    """
    for unit in ['', 'K', 'M', 'G', 'T', 'P', 'E', 'Z']:
        if abs(size) < 1024.0:
            return f'{size:3.1f} {unit}{suffix}'
        size /= 1024.0
    return f'{size:3.1f} Y{suffix}'
