import yaml
from collections import OrderedDict
from os import path as osp


def ordered_yaml():
    """Support OrderedDict for yaml.
    将YAML映射加载为OrderedDicts

    Returns:
        yaml Loader and Dumper.
    """
    try:
        from yaml import CDumper as Dumper
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Dumper, Loader

    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data): # 用表示器（也就是代码里的 repr() 函数）来让你把Python对象转为yaml节点
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node): # 注册一个构造器（也就是代码里的 init() 方法），让你把yaml节点转为Python对象实例
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper


def parse(opt_path, is_train=True):
    """Parse option file.

    Args:
        opt_path (str): Option file path.
        is_train (str): Indicate whether in training or not. Default: True.

    Returns:
        (dict): Options.
    """
    with open(opt_path, mode='r') as f:
        Loader, _ = ordered_yaml() # yaml 文献解析为有序的字典
        opt = yaml.load(f, Loader=Loader) # load yaml文件为有序的字典

    opt['is_train'] = is_train #

    # datasets
    for phase, dataset in opt['datasets'].items():
        # Python 字典(Dictionary) items() 函数以列表返回可遍历的(键, 值) 元组数组。
        # for several datasets, e.g., test_1, test_2
        # opt['datasets'].items() 结果是：[('train', {'name': 'odv-vqa240_train', 'type': 'ODVVQA240Dataset', ...}), (), ..., ()]
        phase = phase.split('_')[0]
        dataset['phase'] = phase # dataset也是个字典，就是 {'name': 'odv-vqa240_train', 'type': 'ODVVQA240Dataset', ...}
        if 'scale' in opt:
            dataset['scale'] = opt['scale']
        # if dataset.get('dataroot_gt') is not None:
        #     dataset['dataroot_gt'] = osp.expanduser(dataset['dataroot_gt'])
            # 在 Unix 上，开头的 ~ 会被环境变量 HOME 代替，如果变量未设置，则通过内置模块 pwd 在 password 目录中查找当前用户的主目录。以 ~user 开头则直接在 password 目录中查找。
        if dataset.get('dataroot_lq') is not None:
            dataset['dataroot_lq'] = osp.expanduser(dataset['dataroot_lq'])

    # paths
    for key, val in opt['path'].items():
        if (val is not None) and ('resume_state' in key or 'pretrain_network' in key):
            opt['path'][key] = osp.expanduser(val)
            print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>res', osp.expanduser(val))
    opt['path']['root'] = opt['path']['experiment_media']
    # opt['path']['root'] = osp.abspath(
    #     osp.join("__file__", osp.pardir, osp.pardir, osp.pardir))
    # os.path.join(os.path.dirname("__file__"),os.path.pardir) # 将文件的当前目录和文件当前目录的上级目录进行合并，取交集
        # 依次为当前，父目录，父父目录的绝对路径
    if is_train:
        experiments_root = osp.join(opt['path']['root'], 'experiments',
                                    opt['name'])
        opt['path']['experiments_root'] = experiments_root
        opt['path']['models'] = osp.join(experiments_root, 'models')
        opt['path']['training_states'] = osp.join(experiments_root,
                                                  'training_states')
        opt['path']['log'] = experiments_root
        opt['path']['visualization'] = osp.join(experiments_root,
                                                'visualization')

        # change some options for debug mode
        if 'debug' in opt['name']:
            if 'val' in opt:
                opt['val']['val_freq'] = 8
            opt['logger']['print_freq'] = 1
            opt['logger']['save_checkpoint_freq'] = 8
    else:  # test
        results_root = osp.join(opt['path']['root'], 'results', opt['name'])
        opt['path']['results_root'] = results_root
        opt['path']['log'] = results_root
        opt['path']['visualization'] = osp.join(results_root, 'visualization')

    return opt


def dict2str(opt, indent_level=1):
    """dict to string for printing options.

    Args:
        opt (dict): Option dict.
        indent_level (int): Indent level. Default: 1.

    Return:
        (str): Option string for printing.
    """
    msg = '\n'
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += ' ' * (indent_level * 2) + k + ':['
            msg += dict2str(v, indent_level + 1)
            msg += ' ' * (indent_level * 2) + ']\n'
        else:
            msg += ' ' * (indent_level * 2) + k + ': ' + str(v) + '\n'
    return msg
