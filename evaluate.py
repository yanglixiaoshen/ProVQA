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
 
    average_best_frame_score_bvqa360()
