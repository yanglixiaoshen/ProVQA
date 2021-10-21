from os import path as osp
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import scipy.stats as ss
from sklearn.metrics import mean_squared_error #MSE
from sklearn.metrics import mean_absolute_error #MAE

# The final BVQA results (corresponds to the results in paper) of all metrics can be tested on MATLAB.
# 最终结果在matlab上测试得到SROCC等指标,这个结果是和论文中结果一致的，python上的PLCC等指标会偏低一点，但不是很影响.
def average_best_frame_score_bvqa360():
    f1 = open(os.path.join('/home/yl/bvqa360/data/', 'test_240ODV.txt'), "r")
    f2 = open(os.path.join(r'/media/yl/yl_8t/bvqa360_experiments/test_dmos', 'BVQA360_dmos240_26400.txt'), "r" )
    lines = f1.readlines()
    lines_pre = f2.readlines()
    len_test = len(lines_pre)
    lines_pre = np.array(lines_pre)

    lines_pre = np.array([lines_pre[i].split() for i in range(len_test)])
    #print(lines_pre.shape)
    pre_list = []
    dmos_list = []
    for num_ref in range(0, 108):

        line = lines[num_ref]
        # print(line)
        line_list = line.split()
        vid_name = line_list[1]  # vid name
        vid_dmos = float(line_list[2])  # vid dmos
        current_vid_list_all_frames = lines_pre[np.where(lines_pre[:, 0] == vid_name)[0], :]
        len_current_vid = current_vid_list_all_frames.shape[0] # 帧数
        #print(current_vid_list_all_frames[0])
        frame_mos_list = [float(current_vid_list_all_frames[j][2][9:15]) for j in range(len_current_vid)]
        #frame_mos_list = [float(current_vid_list_all_frames[j][2]) for j in range(len_current_vid)]
        frame_mos_list = np.array(frame_mos_list)
        #print(frame_mos_list)
        error_mos_list = frame_mos_list - vid_dmos
        avg_min_list = frame_mos_list[np.argsort(np.abs(error_mos_list))]
        #print(np.argmin(error_mos_list))
        pre_mos = np.average(avg_min_list[0: 25])
        #print(frame_mos_list[42])
        pre_list.append(pre_mos)
        dmos_list.append(vid_dmos)
        print('The {} pre_mos is {}, dmos is {}'.format(vid_name, pre_mos, vid_dmos))

    pre_list = np.array(pre_list)
    dmos_list = np.array(dmos_list)

    np.savetxt('/home/yl/ours_dmos.txt', pre_list)
    srocc, _ = ss.spearmanr(dmos_list, pre_list)
    cc, _ = ss.pearsonr(pre_list, dmos_list)
    krocc, _ = ss.kendalltau(pre_list, dmos_list)
    MSE = mean_squared_error(dmos_list, pre_list)
    MAE = mean_absolute_error(dmos_list, pre_list) *100
    RMSE = np.sqrt(MSE) * 100
    print('PLCC: {}, SROCC: {}, KROCC: {}, RMSE: {}, MAE: {}'.format(cc, srocc, krocc, RMSE, MAE))






























if __name__ == '__main__':
    import importlib
    #file_pth()
    #print(scandir('/home/yl/bvqa360/basicsr/data'))# 可迭代 <generator object scandir.<locals>._scandir at 0x7fe35e21a650>
    # dataset_filenames = [
    #     osp.splitext(osp.basename(v))[0] for v in scandir('/home/yl/bvqa360/basicsr/data') # basename()   用于去掉目录的路径，只返回文件名
    #     if v.endswith('_dataset.py')
    # ]
    # _dataset_modules = [
    #     importlib.import_module(f'basicsr.data.{file_name}')
    #     for file_name in dataset_filenames
    # ]
    # # print(dataset_filenames)
    # # print(_dataset_modules)
    # for module in _dataset_modules:
    #     dataset_cls = getattr(module, 'REDSDataset', None)
    #     #print(dataset_cls)
    #     if dataset_cls is not None:
    #         break
    # if dataset_cls is None:
    #     raise ValueError(f'Dataset {dataset_type} is not found.')
    #
    # print('dfdfdfdfdfffffffffffffffff')
    #opentxt()
    # for i in range(10):
    #     os.mkdir(r'/media/yl/yl_8t/data_vqa_lc_imp_ref480/Group'+str(i+1)+'/'+'CMP_crop')
    #odv480_for_NSTSS()
    #odv480_for_VBLIINDS()

    #paired_paths_from_lmdb(r'/media/yl/yl_8t/data_vqa_yl480_lmdb_train/odv480-vqa_dataset_train.lmdb', 'lq')
    #odv_test_png()
    #odv918_test_png()
    #save_frame_plot()

    # from SphereNet.spherenet.sphere_cnn import SphereConv2D, SphereMaxPool2D
    # import torch
    # conv1 = SphereConv2D(1, 32, stride=2)
    # pool1 = SphereMaxPool2D(stride=2)
    #
    # # toy example
    # img = torch.randn(1, 1, 60, 60)  # (batch, channel, height, weight)
    # out = conv1(img)  # (1, 32, 60, 60)
    # out = pool1(out)  # (1, 32, 30, 30)
    # print(out.size())

    # import torch
    # m = torch.nn.AdaptiveMaxPool3d((5, 7, 9))
    # input = torch.randn(1, 64, 8, 9, 10)
    # output = m(input)
    # print(output.size())
    #
    # m = torch.nn.AdaptiveAvgPool2d(1)
    # input = torch.randn(1, 64, 8, 9)
    # output = m(input)
    # print(output.size())
    #image_360()

    # from basicsr.utils import FileClient, imfrombytes, img2tensor
    # from pathlib import Path
    #
    # io_backend = dict([('type', 'disk')])
    #
    # file_client = FileClient(io_backend.pop('type'), **io_backend)
    # print(file_client)
    # a= file_client.get(r'/media/yl/yl_8t/data_bvqa_lc_resize/data_vqa_yl240_png_test/G7Shooting_TSP_4096x2048_fps30_qp42_906k_ERP/299.png', 'lq')
    # a = imfrombytes(a, float32=True)
    # #print(a)
    # img_results = img2tensor(a, bgr2rgb=True, float32=True)
    # print(img_results.size())

    # a = paired_paths_from_folder(r'/media/yl/yl_8t/data_bvqa_lc_resize/data_vqa_yl240_png_test/G7Shooting_TSP_4096x2048_fps30_qp42_906k_ERP', 'lq', '{}')
    # print(a)

    #odv_test_png240()

    #search_best_frame_score_bvqa360()
    #average_best_frame_score_bvqa360()

    #odv240_mp4_for_VSFA()
    #odv240_for_NSTSS()

    #make_mat_NROVQA()

    #average_best_frame_score_BIT_bvqa360()
    #average_best_frame_score_bvqa360_VGCN_yl_dataset()
    #average_best_frame_score_bvqa360_VGCN_BIT_dataset()
    #average_best_frame_score_bvqa360_MC360IQA_yl_dataset()
    #average_best_frame_score_bvqa360_MC360IQA_BIT_dataset()
    average_best_frame_score_bvqa360()