import numpy as np
import random
import torch
from pathlib import Path
from torch.utils import data as data
import os.path as op
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
from basicsr.utils.flow_util import dequantize_flow
from os import path as osp
from basicsr.data.data_util import (paired_paths_from_folder,
                                    paired_paths_from_lmdb,
                                    paired_paths_from_meta_info_file)
import cv2
import os
class ODVVQA240Dataset(data.Dataset):
    """BVQA360 dataset for training.
    """
    def __init__(self, opt): # dataset_opt
        super(ODVVQA240Dataset, self).__init__()
        self.opt = opt
        self.lq_folder = opt['dataroot_lq']
        self.lq_root = Path(opt['dataroot_lq'])
        self.train_info = Path(opt['train_odv240_info_file']) # 3 G1BikingToWork_ERP_3840x2160_fps23.976_qp27_12306k 0.488671931168760 400 240 480

        assert opt['num_frame'] % 1 == 0, (
            f'num_frame should be odd number, but got {opt["num_frame"]}')
        self.num_frame = opt['num_frame'] # 6

        self.neibor_frame = 3
        self.all_neighbor_list = []
        with open(opt['meta_info_file'], 'r') as fin: #  G10BoatInPark_ERP_4096x2048_fps30_qp27_14547k/001.png (240,480,3) 1
            self.keys = [osp.splitext(line.split(' ')[0])[0] for line in fin] # [G10BoatInPark_ERP_4096x2048_fps30_qp27_14547k/001, G10BoatInPark_ERP_4096x2048_fps30_qp27_14547k/002, ...]

        with open(self.train_info) as tr_inf:
            self.train_vid_name = [line.split(' ')[1] for line in tr_inf] # 所有ODV 名字列表
        tr_inf.close()
        with open(self.train_info) as tr_inf1:
            self.train_vid_dmos = [line1.split(' ')[2] for line1 in tr_inf1] # 所有ODV dmos列表
        tr_inf1.close()
        with open(self.train_info) as tr_inf2:
            self.train_vid_frame_num = [line2.split(' ')[3] for line2 in tr_inf2] # 所有ODV frame number列表
        tr_inf2.close()

        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.is_lmdb = False
        if self.io_backend_opt['type'] == 'lmdb':
            self.is_lmdb = True
            self.io_backend_opt['db_paths'] = [self.lq_folder]
            self.io_backend_opt['client_keys'] = ['lq']
            self.paths = paired_paths_from_lmdb(self.lq_folder, 'lq')
            #  [{'lq_path': /G1Aerialxxx/001}, {'lq_path': /G1Aerialxxx/002}, ...]

        self.interval_list = opt['interval_list']
        self.random_reverse = opt['random_reverse']
        interval_str = ','.join(str(x) for x in opt['interval_list'])
        logger = get_root_logger()
        logger.info(f'Temporal augmentation interval list: [{interval_str}]; '
                    f'random reverse is {self.random_reverse}.')

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        key = self.keys[index] # # [G10BoatInPark_ERP_4096x2048_fps30_qp27_14547k/001, G10BoatInPark_ERP_4096x2048_fps30_qp27_14547k/002, ...]

        clip_name = ('/').join(key.split('/')[0:-1]) #   '/vidname'

        clip_name_Non_frefix = clip_name.split('/')[-1] # 'vid_name'

        # determine the neighboring frames
        interval = random.choice(self.interval_list) # 3

        # # ensure not exceeding the borders

        vid_position = np.where(np.array(self.train_vid_name)[:] == clip_name_Non_frefix)[0][0]

        assert vid_position != None, (
            f'Wrong clip name when finding in train_info: {vid_position}')
        vid_dmos = float(self.train_vid_dmos[vid_position])
        vid_frame_num = int(self.train_vid_frame_num[vid_position])
        # print('>>>> index: {}, video name: {}, dmos: {}, frame_num'.format(index, clip_name_Non_frefix, vid_dmos, vid_frame_num))
        start_frame_id = 4
        end_frame_id = vid_frame_num-4

        frame_name = key.split('/')[-1]  # key example: 003
        center_frame_idx = int(frame_name) # 3

        if center_frame_idx < start_frame_id or center_frame_idx > end_frame_id:
            frames_per_batch = np.sort(np.random.randint(start_frame_id, end_frame_id+1, size=self.num_frame)) # 随机取6帧--》排序
        else:
            frames_per_batch_temp = np.sort(np.random.randint(start_frame_id, end_frame_id+1, size=self.num_frame-1)) # 随机取5帧，key帧不能和这5帧重叠
            while center_frame_idx in frames_per_batch_temp:
                frames_per_batch_temp = np.sort(np.random.randint(start_frame_id, end_frame_id + 1, size=self.num_frame - 1))
            frames_per_batch = np.sort(np.append(frames_per_batch_temp, center_frame_idx)) # 6帧排序

        # 6帧以及各自的前后相邻帧 共18帧
        self.all_neighbor_list = [list(range(frames_per_batch[i] - interval, frames_per_batch[i] +  interval + 1, interval)) for i in range(self.num_frame)]

        self.all_neighbor_list = np.array(self.all_neighbor_list).reshape(-1) # 18 frames

        # 检验是否是18帧
        assert self.all_neighbor_list.shape[0] == self.neibor_frame * self.num_frame, (
            f'Wrong length of neighbor list: {self.all_neighbor_list.shape[0]}')

        # 找到 all_neighbor_list 对应于 meta_info_file 中的帧的次序
        if center_frame_idx < start_frame_id:
            index_min = self.all_neighbor_list[0]-center_frame_idx+index # key帧小于最小frame情况
            self.all_neighbor_list1 = index_min+self.all_neighbor_list-self.all_neighbor_list[0]
        elif center_frame_idx > end_frame_id:
            index_max = index - (center_frame_idx-self.all_neighbor_list[-1])   # key帧超出最大frame情况
            self.all_neighbor_list1 = index_max+self.all_neighbor_list-self.all_neighbor_list[-1]
        else:
            self.all_neighbor_list1 = index + self.all_neighbor_list - center_frame_idx  # key帧就在正常范围内情况

        # get the neighboring LQ frames
        img_lqs = []
        ip=0
        for neighbor in self.all_neighbor_list1:
            ip+=1
            if self.is_lmdb:
                img_lq_path = self.paths[neighbor]['lq_path']  # 从path列表中找到对应位置 从字典中提取出对应的帧
            else:
                img_lq_path = self.lq_root / clip_name / f'{neighbor:03d}.png'
            img_bytes = self.file_client.get(img_lq_path, 'lq')
            img_lq = imfrombytes(img_bytes, float32=True)  # BGR format
            img_lqs.append(img_lq)
            #cv2.imwrite(os.path.join('/media/yl/yl_8t/bvqa360_experiments/', str(ip)+'.jpg'), img_lq*255) # BGR 不影响cv2.imwrite()保存 依旧是正确的格式。

        img_results = img2tensor(img_lqs, bgr2rgb = True)  # 训练时候必须将BGR转换为RGB
        img_results = torch.stack(img_results, dim=0)
        #print('>>>>>>>>>>>The indiex is ', index, self.all_neighbor_list, key, img_results.size())
        #print(img_results.size())
        # img_lqs = torch.stack(img_results[0:-1], dim=0)
        # img_gt = img_results[-1]

        # img_lqs: (t, c, h, w)
        # img_gt: (c, h, w)
        # key: str
        #print(">>>> img_lqs1: ", img_results, torch.Tensor([vid_dmos]).size())
        return {'lq': img_results, 'gt': torch.Tensor([vid_dmos]).float(), 'key': key}

    def __len__(self):
        return round(len(self.keys)/self.num_frame)



