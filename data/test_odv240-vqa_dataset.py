from torch.utils import data as data
from torchvision.transforms.functional import normalize
from os import path as osp
from basicsr.data.data_util import (paired_paths_from_folder,
                                    paired_paths_from_lmdb,
                                    paired_paths_from_meta_info_file)
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, imfrombytes, img2tensor
from pathlib import Path
import numpy as np
import torch
import os
import cv2
import random
class ODVVQA240DatasetTest(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and
    GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the
                template excludes the file extension. Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            use_flip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(ODVVQA240DatasetTest, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        self.num_frame = opt['num_frame']
        self.test_info = Path(opt['test_odv480_info_file'])
        self.neibor_frame = 3

        self.lq_folder = opt['dataroot_lq']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            self.paths = paired_paths_from_lmdb(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'])
        elif 'meta_info_file' in self.opt and self.opt[
                'meta_info_file'] is not None:
            self.paths = paired_paths_from_meta_info_file(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'],
                self.opt['meta_info_file'], self.filename_tmpl)
        else:
            self.paths = paired_paths_from_folder(
                self.lq_folder, 'lq',
                self.filename_tmpl)
            #[{'lq_path': '/media/yl/yl_8t/data_bvqa_lc_resize/data_vqa_yl240_png_test/G7Shooting_TSP_4096x2048_fps30_qp42_906k_ERP/001.png'},
            # {'lq_path': '/media/yl/yl_8t/data_bvqa_lc_resize/data_vqa_yl240_png_test/G7Shooting_TSP_4096x2048_fps30_qp42_906k_ERP/002.png'}, ...]


        #print(self.paths)
        with open(self.test_info) as tr_inf:
            self.train_vid_name = [line.split(' ')[1] for line in tr_inf]
        tr_inf.close()
        with open(self.test_info) as tr_inf1:
            self.train_vid_dmos = [line1.split(' ')[2] for line1 in tr_inf1]
        tr_inf1.close()
        with open(self.test_info) as tr_inf2:
            self.train_vid_frame_num = [line2.split(' ')[3] for line2 in tr_inf2]
        tr_inf2.close()



    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        interval = 3


        lq_path = self.paths[index]['lq_path']

        #clip_name = ('/').join(lq_path.split('/')[0:-1])  # /path/vid_name lq_path: /media/yl/yl_8t/data_vqa_yl480_png/G1Aerial_ERP_7680x3840_fps25_qp42_1216k_213
        # print('!!!!!!!!!!!!!!!!!!!!!!!!!!index clip', index, clip_name)
        clip_name_Non_frefix  = ('_').join(osp.splitext(osp.basename(lq_path))[0].split('_')[:-1]) # vid_name  G1Aerial_ERP_7680x3840_fps25_qp42_1216k
        vid_position = np.where(np.array(self.train_vid_name)[:] == clip_name_Non_frefix)[0][0]
        #center_frame_idx = int(osp.splitext(osp.basename(lq_path))[0].split('_')[-1])
        vid_frame_num = int(self.train_vid_frame_num[vid_position])
        vid_dmos = float(self.train_vid_dmos[vid_position])

        start_frame_id = 4
        end_frame_id = vid_frame_num - 4

        key = osp.splitext(osp.basename(lq_path))[0].split('_')[-1]
        frame_name = key # key example: 003
        center_frame_idx = int(frame_name)  # 3

        if center_frame_idx < start_frame_id or center_frame_idx > end_frame_id:
            frames_per_batch = np.sort(np.random.randint(start_frame_id, end_frame_id + 1, size=self.num_frame))
        else:
            frames_per_batch_temp = np.sort(
                np.random.randint(start_frame_id, end_frame_id + 1, size=self.num_frame - 1))
            while center_frame_idx in frames_per_batch_temp:
                frames_per_batch_temp = np.sort(
                    np.random.randint(start_frame_id, end_frame_id + 1, size=self.num_frame - 1))
            frames_per_batch = np.sort(np.append(frames_per_batch_temp, center_frame_idx))
        # frames_per_batch_temp = np.sort(np.random.randint(start_frame_id, end_frame_id + 1, size=self.num_frame - 1))
        # frames_per_batch = np.sort(np.append(frames_per_batch_temp, center_frame_idx))

        # self.f.writelines('{} {}'.format(clip_name_Non_frefix, vid_position))
        self.all_neighbor_list = [
            list(range(frames_per_batch[i] - interval, frames_per_batch[i] + interval + 1, interval)) for i in
            range(self.num_frame)]
        # print(self.all_neighbor_list)
        self.all_neighbor_list = np.array(self.all_neighbor_list).reshape(-1)  # 30 frames
        #print(self.all_neighbor_list)
        #print(self.lq_folder)

        # left_frame_pth = os.path.join('/media/yl/yl_8t/data_vqa_yl480_png_test', clip_name_Non_frefix+'_' + '{:0>3}'.format(str(left_frame_idx)) + '.png')
        # right_frame_pth = os.path.join('/media/yl/yl_8t/data_vqa_yl480_png_test', clip_name_Non_frefix+'_' +
        #                              '{:0>3}'.format(str(right_frame_idx)) + '.png')

        #print(left_frame_pth, lq_path, right_frame_pth)
        img_lqs = []
        for neighbor in self.all_neighbor_list:
            #print('###############The neighbor is:', neighbor)

            img_lq_path = os.path.join(self.lq_folder, clip_name_Non_frefix+'_' +
                                     '{:0>3}'.format(str(neighbor)) + '.png')
            #print(img_lq_path)
            #print('>>>>>>>>>>>>>>>img_path', img_lq_path)
            img_bytes = self.file_client.get(img_lq_path, 'lq')
            #print('>>>>>>>>>>>>>>>img_bytes: ', img_bytes)
            img_lq = imfrombytes(img_bytes, float32=True)  # BGR format
            img_lqs.append(img_lq)



        img_results = img2tensor(img_lqs, bgr2rgb=True, float32=True)
        img_results = torch.stack(img_results, dim=0)
        # TODO: color space transform
        # BGR to RGB, HWC to CHW, numpy to tensor




        # # normalize
        # if self.mean is not None or self.std is not None:
        #     normalize(img_lq, self.mean, self.std, inplace=True)
        #     normalize(img_gt, self.mean, self.std, inplace=True)

        return {
            'lq': img_results,
            'gt': torch.Tensor([vid_dmos]).float(),
            'lq_path': lq_path
        }

    def __len__(self):
        return len(self.paths)


# test image目录里只放108张测试图像 108个视频的某一帧  然后 每次取一个视频 在get_item中指定每个视频的取帧策略
# 1. 随机6*3帧 取多次平均 2. 按顺序取6*3， 共 n_frame//6 次，最后取平均