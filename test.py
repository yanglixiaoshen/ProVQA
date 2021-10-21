import logging
import torch
from os import path as osp

#from basicsr.data import create_dataloader, create_dataset
#from basicsr.models import create_model
from basicsr.train import parse_options
from basicsr.utils import (get_env_info, get_root_logger, get_time_str,
                           make_exp_dirs)
from basicsr.utils.options import dict2str
from data import create_dataloader, create_dataset
from model import create_model

def main():
    # parse options, set distributed setting, set ramdom seed
    opt = parse_options(is_train=False)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # mkdir and initialize loggers
    make_exp_dirs(opt)
    log_file = osp.join(opt['path']['log'],
                        f"test_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(
        logger_name='test_bvqa360', log_level=logging.INFO, log_file=log_file)
    #logger.info(get_env_info())
    logger.info(dict2str(opt))

    # create test dataset and dataloader
    test_loaders = []
    for phase, dataset_opt in sorted(opt['datasets'].items()):
        test_set = create_dataset(dataset_opt)
        test_loader = create_dataloader(
            test_set,
            dataset_opt,
            num_gpu=opt['num_gpu'],
            dist=opt['dist'],
            sampler=None,
            seed=opt['manual_seed'])
        logger.info(
            f"Number of test images in {dataset_opt['name']}: {len(test_set)}")
        test_loaders.append(test_loader)

    # create model
    model = create_model(opt)

    # tt
    for test_loader in test_loaders:
        test_set_name = test_loader.dataset.opt['name']
        logger.info(f'Testing {test_set_name}...')
        model.validation(
            test_loader,
            current_iter=opt['name'],
            tb_logger=None,
            save_img=opt['val']['save_img'])
        # tt
    ##


if __name__ == '__main__':
    main()

"""
PYTHONPATH="./:${PYTHONPATH}" \
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 \
python -m torch.distributed.launch --nproc_per_node=7 --master_port=4324 test.py -opt ../options/test/ml_edvr/ali/0309_our_test_qunliang_enhanced_train_raw_and_lq_to_finetune_edvr.yml --launcher pytorch
"""

"""
PYTHONPATH="./:${PYTHONPATH}" \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python -m torch.distributed.launch --nproc_per_node=8 --master_port=4321 test.py -opt ../options/test/ml_edvr/ml_test_EDVR_L_deblurcomp_REDS.yml --launcher pytorch
"""

""" test esrgan L1
PYTHONPATH="./:${PYTHONPATH}" \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python -m torch.distributed.launch --nproc_per_node=8 --master_port=4321 test.py \
-opt ../options/test/ESRGAN/test_bvqa360.yml --launcher pytorch


CUDA_VISIBLE_DEVICES=2 \
python test.py \
-opt ../options/test/ESRGAN/test_bvqa360.yml
"""