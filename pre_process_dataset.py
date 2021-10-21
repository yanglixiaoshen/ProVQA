import os
import numpy as np
import shutil
#from os import path as osp
import cv2
import lmdb
import sys
from multiprocessing import Pool
from os import path as osp
from tqdm import tqdm
from utils.ws_ssim import ws_ssim
from PIL import Image
########################
##### Dataset path #####
########################

org_dir = r'/media/yl/yl_8t/data_bvqa_lc_resize/data_vqa_lc_LR240' # lc 原始数据库ERP格式降低分辨率后的YUV 480 240, 然后ERP 转换成三种不同格式的 180个
imp_dir = r'/media/yl/yl_8t/data_bvqa_lc_resize/data_vqa_lc_imp_ref240' # 各种格式的ref视频HEVC压缩后的视频 540个 还没有经过重映射回ERP格式
#imp_erp_dir = r'/media/yl/yl_8t/data_vqa_yl720'
imp_erp_dir = r'/media/yl/yl_8t/data_bvqa_lc_resize/data_vqa_yl240' # 经过两种失真后的540个ODV
com_cmp_dir = r'/media/yl/yl_8t/com_cmp'
dataset_utils = r'/home/yl/bvqa360/data/' # 存放各种文件信息 dmos name等txt

orig_data = r'/media/liutie/pami_yl/DatabaseVideoYUV'  # LC 大硬盘里的最原始的数据库 yuv
lr_data = r'/media/liutie/pami_yl/data_vqa_lc_LR'  # 处理第一步

host_dir = r'/media/yl/yl_8t/'
log_dir = r'/media/yl/yl_8t/data_bvqa_lc_resize/data_vqa_lc_logbin240/log/' # H.265压缩编码后信息储存的地方
bin_dir = r'/media/yl/yl_8t/data_bvqa_lc_resize/data_vqa_lc_logbin240/bin/'

#########################################################################################################################################################
##### Create the lc ODV-VQA dataset (format: YUV; number: 540; resolution: (480, 240); Projection: ERP, RCMP, TSP; Compression levels: QP=27,37,42) #####
#########################################################################################################################################################


# 1. 原始ODV降分辨率： 将60个原始ODV (yuv格式，orig_data/Reference目录下：10个Group, 每个Group里面有6个原始无失真参考ODV) 用ffmpeg降低分辨率到480×240，
#    放入到 lr_data 目录下 （另一个8T lc 硬盘 中），用SCP命令将所有将分辨率后的ODV传送到我的硬盘 （org_dir/Reference 目录下）。

def org_lr():
    for group in os.listdir(orig_data):
        if not os.path.exists(os.path.join(lr_data, group)):
            os.mkdir(os.path.join(lr_data, group))
        for obj in os.listdir(os.path.join(orig_data, group)):
            #print(obj)
            if os.path.isdir(os.path.join(orig_data, group, obj)): #  reference directory
                print("it's a directory {}".format(obj))
                if not os.path.exists(os.path.join(lr_data, group, obj)):
                    os.mkdir(os.path.join(lr_data, group, obj))
                for ref in os.listdir(os.path.join(orig_data, group, obj)):
                    res = ref.split('_')[1]
                    res_lr = '480x240'
                    org_ref_video = os.path.join(orig_data, group, obj, ref)
                    lr_ref_video = os.path.join(lr_data, group, obj, ref)
                    cmd1 = r'ffmpeg -f rawvideo -s {} -pix_fmt yuv420p -i {} -s {} -c:v rawvideo {}'.format(res, org_ref_video, res_lr, lr_ref_video)
                    os.system(cmd1)
            else:
                print('it is a special file(socket,FIFO,device file)')

# 2. 加入映射失真： 将所有60个原始参考ODV 使用360tools转化为CMP TSP ERP三种格式，共3*60=180个，每组18个，还放在org_dir目录下；
#    执行程序：将.py置于/home/yl/360tools/bin 下进行映射转换。
def convert_360():

    for group in os.listdir(org_dir): # Group1...
        if not os.path.exists(os.path.join(imp_dir, group)):
            os.mkdir(os.path.join(imp_dir, group))
        # if not os.path.exists(os.path.join(org_dir, group, 'CMP')):
        # 	os.mkdir(os.path.join(org_dir, group, 'CMP'))
        ref_dir = os.path.join(org_dir, group, 'Reference')
        for ref_video in os.listdir(ref_dir):
            ref_video_sp = ref_video.split('_')
            erp_name = '_'.join([ref_video_sp[0], 'ERP']+ref_video_sp[1:])
            tsp_name = '_'.join([ref_video_sp[0], 'TSP']+ref_video_sp[1:])
            cmp_name = '_'.join([ref_video_sp[0], 'RCMP']+ref_video_sp[1:])
            #rcmp_name = '_'.join([ref_video_sp[0], 'RCMP']+ref_video_sp[1:])

            ref_video_pth = os.path.join(ref_dir, ref_video)
            erp_video_pth = os.path.join(os.path.join(org_dir, group), erp_name)
            tsp_video_pth = os.path.join(os.path.join(org_dir, group), tsp_name)
            cmp_video_pth = os.path.join(os.path.join(org_dir, group), cmp_name)
            #rcmp_video_pth = os.path.join(os.path.join(org_dir, group), rcmp_name)
            cmd1 = './360tools_conv -i {} -w 480 -h 240 -x 1 -o {} -l 480 -m 240 -y 1 -f 7'.format(ref_video_pth, tsp_video_pth)
            cmd2 = './360tools_conv -i {} -w 480 -h 240 -x 1 -o {} -l 480 -m 360 -y 1 -f 3'.format(ref_video_pth, cmp_video_pth)
            #cmd3 = './360tools_conv -i {} -w 1280 -h 960 -x 1 -o {} -l 1280 -m 960 -y 1 -f 13'.format(cmp_video_pth, rcmp_video_pth)
            shutil.copyfile(ref_video_pth, erp_video_pth)

            os.system(cmd1)
            os.system(cmd2)
            #os.system(cmd3)

# 3. 加入压缩失真：将180个已经加入映射失真的ODV每一个都加入QP=27, 37, 42的压缩失真， 存放在 imp_dir 目录下。 10个Group,每个Group 54个ODV。
    # 新的 480×240 ODV, 180个 ERP TSP RCMP失真的所有yuv编成.sh
    # 多线程编码：目录在 /media/yl/yl_8t/Encoder_exdata_VQA_180Seqs_RA_Schedule
    # 首先删掉Encoder_#1-16 所有目录，改写run.info文件，使他成为0 1 0 （视频全部编码好就成为540 1 0）
    # 然后把rename后的.sh文件重命名为run.sh （本程序：vqa_lc_rename240.txt-> run.sh）
    # 最后命令行跑 python run_multi_threads.py 开始多线程编码

def comp_vqa_lc_HM_rename_240():
    # 写之前，先检验文件是否存在，存在就删掉
    if os.path.exists(os.path.join(host_dir, 'vqa_lc_rename240.txt')):
        os.remove(os.path.join(host_dir, 'vqa_lc_rename240.txt'))
    num_frames = np.loadtxt(os.path.join(dataset_utils, 'vqa_frameNums.txt'))
    num_frames = num_frames.reshape((10, 18))
    f = open(os.path.join(host_dir, 'vqa_lc_rename240.txt'), 'w')
    g = -1
    for gg in range(10):
        group = 'Group' + str(gg + 1)
        if not os.path.exists(os.path.join(log_dir, group)):
            os.mkdir(os.path.join(log_dir, group))
        if not os.path.exists(os.path.join(bin_dir, group)):
            os.mkdir(os.path.join(bin_dir, group))
        g += 1
        num_group_frame = num_frames[g]
        index = -1
        for obj in os.listdir(os.path.join(org_dir, group)):
            if os.path.isfile(os.path.join(org_dir, group, obj)):
                if obj[0] == 'G':
                    index += 1
                    obj_splt = obj.split('_')
                    obj_proj = obj_splt[1]
                    if obj_proj == 'ERP':
                        W, H = 480, 240
                    elif obj_proj == 'RCMP':
                        W, H = 480, 360
                    elif obj_proj == 'TSP':
                        W, H = 480, 240

                    nf = num_group_frame[index]

                    org_file = os.path.join(org_dir, group, obj)

                    obj_name_splt = obj.split('.')[0:-1]
                    obj_name_splt = '.'.join(obj_name_splt)
                    print(obj_name_splt)
                    for qp_num in [27, 37, 42]:
                        print(obj_name_splt)
                        qp_num_str = 'qp' + str(qp_num)
                        obj_yuv = obj_name_splt + '_' + qp_num_str + '.yuv'
                        obj_log = obj_name_splt + '_' + qp_num_str + '.txt'
                        obj_bin = obj_name_splt + '_' + qp_num_str + '.bin'
                        # print(obj_yuv)

                        com_file = os.path.join(imp_dir, group, obj_yuv)
                        log_file = os.path.join(log_dir, group, obj_log)
                        bin_file = os.path.join(bin_dir, group, obj_bin)

                        vqa_coding_cmd = './TAppEncoderStatic -c encoder_randomaccess_main.cfg -c encoder_others.cfg -i {} -wdt {} -hgt {} -fr 30 -f {} -q {} -b {} -o {} >{}\n'.format(
                            org_file, W, H, int(nf), qp_num, bin_file, com_file, log_file)

                        f.writelines(vqa_coding_cmd)
            else:
                continue
    f.close()

# 4. 重映射： 将压缩后的540个ODV重新映射回ERP格式，将所有540个ODV分成10组放入到 imp_erp_dir 目录中。
#    运行程序：将.py置于/home/yl/360tools/bin 下进行映射转换。

def com_yuv_convert_yl240():
    f = open(os.path.join(dataset_utils, 'train.txt'),"r")
    lines = f.readlines()      #读取全部内容 ，并以列表方式返回
    namelist = []
    for num_ref in range(0, 540):
        line = lines[num_ref]
        line_list = line.split() # string to list
        name = line_list[3]
        filename = os.path.splitext(name.split(os.sep)[-1])[0] # 去路径，去后缀，G1AbandonedKingdom_ERP_7680x3840_fps30_qp27_45406k
        namelist += [filename]
    print(namelist)

    for vid in range(0, 540):
        vid_name = namelist[vid]
        vid_name_spl = vid_name.split('_')
        group_index = vid_name_spl[0][1]
        group_name = 'Group' + group_index
        if vid>=(540-54):
            group_name = 'Group' + str(10)
        if not os.path.exists(os.path.join(imp_erp_dir, group_name)):
            os.mkdir(os.path.join(imp_erp_dir, group_name))
        vid_pth = os.path.join(imp_dir, group_name)
        if vid_name_spl[1] == 'ERP':
            vid_org_name = ('_').join(vid_name_spl[:-1])
            vid_orh_pth = os.path.join(vid_pth, vid_org_name+'.yuv')
            vid_out_pth = os.path.join(imp_erp_dir, group_name, vid_name+'.yuv')
            shutil.copyfile(vid_orh_pth, vid_out_pth)
            print('Successfully !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        elif vid_name_spl[1] == 'TSP':
            vid_org_name = ('_').join(vid_name_spl[:-2])
            vid_orh_pth = os.path.join(vid_pth, vid_org_name+'.yuv')
            vid_out_pth = os.path.join(imp_erp_dir, group_name, vid_name+'.yuv')
            cmd1 = './360tools_conv -i {} -w 480 -h 240 -x 1 -o {} -l 480 -m 240 -y 1 -f 8'.format(vid_orh_pth, vid_out_pth)
            os.system(cmd1)
            print('Successfully !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        elif vid_name_spl[1] == 'RCMP':
            vid_org_name = ('_').join(vid_name_spl[:-2])
            vid_orh_pth = os.path.join(vid_pth, vid_org_name+'.yuv')
            vid_out_pth = os.path.join(imp_erp_dir, group_name,  vid_name+'.yuv')
            cmd2 = './360tools_conv -i {} -w 480 -h 360 -x 1 -o {} -l 480 -m 240 -y 1 -f 4'.format(vid_orh_pth, vid_out_pth)
            os.system(cmd2)
            print('Successfully !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        else:
            continue

########## Apendix of 360tools ##############
# ./360tools_conv
# -i argument should be set
# < Usage >
#   -c, --config [STRING] (optional)
#     : config file name
#   -i, --input [STRING]
#     : input file name
#   -o, --output [STRING]
#     : output file name
#   -w, --width [INTEGER]
#     : width of input image
#   -h, --height [INTEGER]
#     : height of input image
#   -f, --convfmt [INTEGER]
#     : converting format
#         0:  ERP  to CPP
#         1:  ERP  to ISP
#         2:  ISP  to ERP
#         3:  ERP  to CMP
#         4:  CMP  to ERP
#         5:  ERP  to OHP
#         6:  OHP  to ERP
#         7:  ERP  to TSP
#         8:  TSP  to ERP
#         9:  ERP  to SSP
#         10: SSP  to ERP
#         11: ISP  to RISP
#         12: RISP to ISP
#         13: CMP  to RCMP
#         14: RCMP to CMP
#         15: OHP  to ROHP
#         16: ROHP to OHP
#         21: ERP  to RISP
#         22: RISP to ERP
#         25: ERP  to COHP
#         26: COHP to ERP
#         31: CPP  to ERP
#         32: CPP  to ISP
#         33: CPP  to CMP
#         34: CPP  to OHP
#         35: CPP  to TSP
#         36: CPP  to SSP
#
#   -l, --out_width [INTEGER]
#     : width of output image. if not set, this is same value with width of input image
#   -m, --out_height [INTEGER]
#     : height of output image.  if not set, this is same value with height of input image
#   -n, --frmnum [INTEGER] (optional)
#     : number of frames to be converted
#   -a, --align [INTEGER] (optional)
#     : vertical align
#   -u, --no_pad [FLAG] (optional)
#     : turn off padding
#   -x, --cs_in [INTEGER] (optional)
#     : Input Color Space
#          1: YUV420 8-bit
#          2: YUV420 10-bit
#   -y, --cs_out [INTEGER] (optional)
#     : Output Color Space
#          1: YUV420 8-bit
#          2: YUV420 10-bit
#   -p, --pitch [INTEGER] (optional)
#     : TSP pitch angle [0,180] default is 90
#   -t, --yaw [INTEGER] (optional)
#     : TSP yaw angle [0,360] default is 0


#########################################################################################################################################################
##### Prepare the train (LMDB) and test (PNG) set for BVQA360 #####
#########################################################################################################################################################

####################################
##### Dataset path for BVQA360 #####
####################################

dataset_240= r'/media/yl/yl_8t/data_bvqa_lc_resize/data_vqa_yl240' # 10组视频 每组54个失真的 共540个
png240_dir = r'/media/yl/yl_8t/data_bvqa_lc_resize/data_vqa_yl240_png' # 所有540个目录 每个目录有n帧png图像 每个目录帧数不一样
odv_train240_png = r'/media/yl/yl_8t/data_bvqa_lc_resize/data_vqa_yl240_png_train' # train 训练集 432个目录的png
odv_test_png240 = r'/media/yl/yl_8t/data_bvqa_lc_resize/data_vqa_yl240_png_test' # test 测试集 108个目录的png
new_data_png240 = r'/media/yl/yl_8t/data_bvqa_lc_resize/data_all_test_png' # 所有测试的视频所有帧全部放在这一个目录下---》这个是用于test时候的输入图像
odv_ref_png240 = r'/media/yl/yl_8t/data_bvqa_lc_resize/data_vqa_yl240_png_ref' # 十个Group, 每个Group中有6个reference ODV目录，每个目录下有n帧 PNG图像
odvLmdb_train_pth2 = r'/media/yl/yl_8t/data_bvqa_lc_resize/data_vqa_yl240_lmdb_train/odv240-vqa_dataset_train.lmdb'

# 这4个文件都是按照train 432 test 108 共540个 顺序排列的 存放视频信息 包括： 名字 dmos 帧数 w h num_frame
video_trates_info = r'/home/yl/bvqa360/data/train_test_240ODV_all_info.txt' # 0 G1AbandonedKingdom_ERP_7680x3840_fps30_qp27_45406k 0.414890202945909 300 240 480
video_tratest_NSTSS_info = r'/home/yl/video_file_names_odv240.txt' # G1AbandonedKingdom_ERP_7680x3840_fps30_qp27_45406k 480 240 300
video_tratest_dmos = r'/home/yl/bvqa360/data/train_test_240ODV_dmos.txt' # 0.414890202945909, 0.490389955570886, 0.566788675185082, ..., 0.620770373788080
video_tratest_name = r'/home/yl/bvqa360/data/train_test_240ODV_name.txt' # G1AbandonedKingdom_ERP_7680x3840_fps30_qp27_45406k, G1AbandonedKingdom_ERP_7680x3840_fps30_qp37_9283k,...,



# ID是reference 序号 0~59
odv_train_id = [0, 1 ,2, 5 ,6 ,7, 8 ,9, 10, 11, 13, 14, 15, 16, 18, 20, 22, 24, 25, 26, 27, 29, 31 ,32, 33, 34, 35, 36, 37, 38, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 59]
odv_test_id = [3, 4 ,12, 17, 19 ,21 ,23 ,28, 30, 39, 40, 58]

def scandir(dir_path, suffix=None, recursive=False, full_path=False):
    """Scan a directory to find the interested files.
    suffix: 搜索文件的后缀 recursive：递归搜索， full_path: 搜索包括目录
    # 在dir_pth目录下 可以递归或者不递归的生成一个文件迭代器，包含所有文件名，不包含目录。
    Args:
        dir_path (str): Path of the directory.
        suffix (str | tuple(str), optional): File suffix that we are
            interested in. Default: None.
        recursive (bool, optional): If set to True, recursively scan the
            directory. Default: False. 递归查找文件或者目录
        full_path (bool, optional): If set to True, include the dir_path.
            Default: False.

    Returns:
        A generator for all the interested files with relative pathes.
    """

    if (suffix is not None) and not isinstance(suffix, (str, tuple)):
        raise TypeError('"suffix" must be a string or tuple of strings')

    root = dir_path

    def _scandir(dir_path, suffix, recursive):
        for entry in os.scandir(dir_path): # generator 可迭代生成器
            #print(entry) # <DirEntry 'meta_info_REDS4_test_GT.txt'>

            #print(entry.name) # meta_info_REDS4_test_GT.txt
            if not entry.name.startswith('.') and entry.is_file():
                if full_path:
                    return_path = entry.path
                else:
                    #print(entry.path) # /home/yl/bvqa360/basicsr/data/transforms.py
                    return_path = osp.relpath(entry.path, root) # os.path.relpath()Python中的方法用于从当前工作目录或给定目录获取到给定路径的相对文件路径。
                    #print(return_path) # transforms.py

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


# 1. 将 dataset_240 .yuv 视频通过ffmpeg转换为 .png 图像；png240_dir 包含540个目录，每个目录中有每一帧的png图像。

def yuv2png_240():
    for group in os.listdir(dataset_240):
        group_pth = os.path.join(dataset_240, group)
        for vid in os.listdir(group_pth):
            name, ext = os.path.splitext(vid)
            vid_pth = os.path.join(dataset_240, group, vid)
            if not os.path.exists(os.path.join(png240_dir, name)):
                os.mkdir(os.path.join(png240_dir, name))
            out_dir = os.path.join(png240_dir, name, '%3d.png')
            cmd = 'ffmpeg -f rawvideo -pixel_format yuv420p -s 480x240 -i {} {}'.format(vid_pth, out_dir)
            os.system(cmd)

def yuv2png_ref_240():
    for group in os.listdir(org_dir):
        if not os.path.exists(os.path.join(odv_ref_png240, group)):
            os.mkdir(os.path.join(odv_ref_png240, group))
        for obj in os.listdir(os.path.join(org_dir, group)):
            #print(obj)
            if os.path.isdir(os.path.join(org_dir, group, obj)): #  reference directory
                print("it's a directory {}".format(obj))
                if not os.path.exists(os.path.join(odv_ref_png240, group, obj)):
                    os.mkdir(os.path.join(odv_ref_png240, group, obj))
                for ref in os.listdir(os.path.join(org_dir, group, obj)):
                    vid_pth = os.path.join(org_dir, group, obj, ref)
                    ref_new = os.path.splitext(ref)[0]
                    if not os.path.exists(os.path.join(odv_ref_png240, group, obj, ref_new)):
                        os.mkdir(os.path.join(odv_ref_png240, group, obj, ref_new))
                    out_dir = os.path.join(odv_ref_png240, group, obj, ref_new, '%3d.png')
                    cmd = 'ffmpeg -f rawvideo -pixel_format yuv420p -s 480x240 -i {} {}'.format(vid_pth, out_dir)
                    os.system(cmd)


    #
    #
    #
    # for group in os.listdir(dataset_240):
    #     group_pth = os.path.join(dataset_240, group)
    #     for vid in os.listdir(group_pth):
    #         name, ext = os.path.splitext(vid)
    #         vid_pth = os.path.join(dataset_240, group, vid)
    #         if not os.path.exists(os.path.join(png240_dir, name)):
    #             os.mkdir(os.path.join(png240_dir, name))
    #         out_dir = os.path.join(png240_dir, name, '%3d.png')
    #         cmd = 'ffmpeg -f rawvideo -pixel_format yuv420p -s 480x240 -i {} {}'.format(vid_pth, out_dir)
    #         os.system(cmd)

# 将png240_dir中的图像 分配train到odv_train240_png
def shutil_train_test_png():
    f = open(os.path.join(dataset_utils, 'train.txt'), "r")
    lines = f.readlines()  # 读取全部内容 ，并以列表方式返回

    vid_all_list = []
    keys_all_list = []
    # num = 1
    for num_ref in range(0, 540):
        line = lines[num_ref]
        # print(line)
        line_list = line.split()
        vid_id = line_list[0]
        vid_id = int(vid_id)
        # print(vid_id)
        if vid_id in odv_train_id:
            # print('!!!!!!!!!!')
            # num+=1
            # print('the train number is {}'.format(num))
            vid_name = os.path.split(line_list[3])[-1].split('.yuv')[0]

            # print(vid_name)
            folder_path = os.path.join(odv_train240_png, vid_name)
            # if not os.path.exists(folder_path):
            #     os.mkdir(folder_path)
            shutil.copytree(os.path.join(png240_dir, vid_name), folder_path)

# 将png240_dir中的图像 分配test到odv_test240_png
def shutil_test_png():
    f = open(os.path.join(dataset_utils, 'train.txt'), "r")
    lines = f.readlines()  # 读取全部内容 ，并以列表方式返回

    vid_all_list = []
    keys_all_list = []
    # num = 1
    for num_ref in range(0, 540):
        line = lines[num_ref]
        # print(line)
        line_list = line.split()
        vid_id = line_list[0]
        vid_id = int(vid_id)
        # print(vid_id)
        if vid_id in odv_test_id:
            # print('!!!!!!!!!!')
            # num+=1
            # print('the train number is {}'.format(num))
            vid_name = os.path.split(line_list[3])[-1].split('.yuv')[0]

            # print(vid_name)
            folder_path = os.path.join(odv_test_png240, vid_name)
            # if not os.path.exists(folder_path):
            #     os.mkdir(folder_path)
            shutil.copytree(os.path.join(png240_dir, vid_name), folder_path)



# 2. 为训练BVQA360网络，制作train set的LMDB文件

def create_lmdb_for_odv_train480_new():
    img_path_list, keys = prepare_keys_reds(odv_train240_png)

    make_lmdb_from_imgs(odv_train240_png, odvLmdb_train_pth2, img_path_list, keys, multiprocessing_read=True)



def prepare_keys_reds(folder_path):
    """Prepare image path list and keys for REDS dataset.
    Args:
        folder_path (str): Folder path.
    Returns:
        list[str]: Image path list.
        list[str]: Key list.
    """
    print('Reading image path list ...')
    img_path_list = sorted(
        list(scandir(folder_path, suffix='png', recursive=True)))
    keys = [v.split('.png')[0] for v in img_path_list]  # example: 000/00000000
    print(keys)
    return img_path_list, keys


def make_lmdb_from_imgs(data_path,
                        lmdb_path,
                        img_path_list,
                        keys,
                        batch=5000,
                        compress_level=1,
                        multiprocessing_read=False,
                        n_thread=40,
                        map_size=None):
    """Make lmdb from images.
    Contents of lmdb. The file structure is:
    example.lmdb
    ├── data.mdb
    ├── lock.mdb
    ├── meta_info.txt
    The data.mdb and lock.mdb are standard lmdb files and you can refer to
    https://lmdb.readthedocs.io/en/release/ for more details.
    The meta_info.txt is a specified txt file to record the meta information
    of our datasets. It will be automatically created when preparing
    datasets by our provided dataset tools.
    Each line in the txt file records 1)image name (with extension),
    2)image shape, and 3)compression level, separated by a white space.
    For example, the meta information could be:
    `000_00000000.png (720,1280,3) 1`, which means:
    1) image name (with extension): 000_00000000.png;
    2) image shape: (720,1280,3);
    3) compression level: 1
    We use the image name without extension as the lmdb key.
    If `multiprocessing_read` is True, it will read all the images to memory
    using multiprocessing. Thus, your server needs to have enough memory.
    Args:
        data_path (str): Data path for reading images.
        lmdb_path (str): Lmdb save path.
        img_path_list (str): Image path list.
        keys (str): Used for lmdb keys.
        batch (int): After processing batch images, lmdb commits.
            Default: 5000.
        compress_level (int): Compress level when encoding images. Default: 1.
        multiprocessing_read (bool): Whether use multiprocessing to read all
            the images to memory. Default: False.
        n_thread (int): For multiprocessing.
        map_size (int | None): Map size for lmdb env. If None, use the
            estimated size from images. Default: None
    """

    assert len(img_path_list) == len(keys), (
        'img_path_list and keys should have the same length, '
        f'but got {len(img_path_list)} and {len(keys)}')
    print(f'Create lmdb for {data_path}, save to {lmdb_path}...')
    print(f'Totoal images: {len(img_path_list)}')
    if not lmdb_path.endswith('.lmdb'):
        raise ValueError("lmdb_path must end with '.lmdb'.")
    if osp.exists(lmdb_path):
        print(f'Folder {lmdb_path} already exists. Exit.')
        sys.exit(1)

    if multiprocessing_read:
        # read all the images to memory (multiprocessing)
        dataset = {}  # use dict to keep the order for multiprocessing
        shapes = {}
        print(f'Read images with multiprocessing, #thread: {n_thread} ...')
        pbar = tqdm(total=len(img_path_list), unit='image')

        def callback(arg):
            """get the image data and update pbar."""
            key, dataset[key], shapes[key] = arg
            pbar.update(1)
            pbar.set_description(f'Read {key}')

        pool = Pool(n_thread)
        for path, key in zip(img_path_list, keys):
            pool.apply_async(
                read_img_worker,
                args=(osp.join(data_path, path), key, compress_level),
                callback=callback)
        pool.close()
        pool.join()
        pbar.close()
        print(f'Finish reading {len(img_path_list)} images.')

    # create lmdb environment
    if map_size is None:
        # obtain data size for one image
        img = cv2.imread(
            osp.join(data_path, img_path_list[0]), cv2.IMREAD_UNCHANGED)
        _, img_byte = cv2.imencode(
            '.png', img, [cv2.IMWRITE_PNG_COMPRESSION, compress_level])
        data_size_per_img = img_byte.nbytes
        print('Data size per image is: ', data_size_per_img)
        data_size = data_size_per_img * len(img_path_list)
        map_size = data_size * 10

    env = lmdb.open(lmdb_path, map_size=map_size)

    # write data to lmdb
    pbar = tqdm(total=len(img_path_list), unit='chunk')
    txn = env.begin(write=True)
    txt_file = open(osp.join(lmdb_path, 'meta_info.txt'), 'w')
    for idx, (path, key) in enumerate(zip(img_path_list, keys)):
        pbar.update(1)
        pbar.set_description(f'Write {key}')
        key_byte = key.encode('ascii')
        if multiprocessing_read:
            img_byte = dataset[key]
            h, w, c = shapes[key]
        else:
            _, img_byte, img_shape = read_img_worker(
                osp.join(data_path, path), key, compress_level)
            h, w, c = img_shape

        txn.put(key_byte, img_byte)
        # write meta information
        txt_file.write(f'{key}.png ({h},{w},{c}) {compress_level}\n')
        if idx % batch == 0:
            txn.commit()
            txn = env.begin(write=True)
    pbar.close()
    txn.commit()
    env.close()
    txt_file.close()
    print('\nFinish writing lmdb.')

def read_img_worker(path, key, compress_level):
    """Read image worker.
    Args:
        path (str): Image path.
        key (str): Image key.
        compress_level (int): Compress level when encoding images.
    Returns:
        str: Image key.
        byte: Image byte.
        tuple[int]: Image shape.
    """

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img.ndim == 2:
        h, w = img.shape
        c = 1
    else:
        h, w, c = img.shape
    _, img_byte = cv2.imencode('.png', img,
                               [cv2.IMWRITE_PNG_COMPRESSION, compress_level])
    return (key, img_byte, (h, w, c))


class LmdbMaker():
    """LMDB Maker.
    Args:
        lmdb_path (str): Lmdb save path.
        map_size (int): Map size for lmdb env. Default: 1024 ** 4, 1TB.
        batch (int): After processing batch images, lmdb commits.
            Default: 5000.
        compress_level (int): Compress level when encoding images. Default: 1.
    """

    def __init__(self,
                 lmdb_path,
                 map_size=1024**4,
                 batch=5000,
                 compress_level=1):
        if not lmdb_path.endswith('.lmdb'):
            raise ValueError("lmdb_path must end with '.lmdb'.")
        if osp.exists(lmdb_path):
            print(f'Folder {lmdb_path} already exists. Exit.')
            sys.exit(1)

        self.lmdb_path = lmdb_path
        self.batch = batch
        self.compress_level = compress_level
        self.env = lmdb.open(lmdb_path, map_size=map_size)
        self.txn = self.env.begin(write=True)
        self.txt_file = open(osp.join(lmdb_path, 'meta_info.txt'), 'w')
        self.counter = 0

    def put(self, img_byte, key, img_shape):
        self.counter += 1
        key_byte = key.encode('ascii')
        self.txn.put(key_byte, img_byte)
        # write meta information
        h, w, c = img_shape
        self.txt_file.write(f'{key}.png ({h},{w},{c}) {self.compress_level}\n')
        if self.counter % self.batch == 0:
            self.txn.commit()
            self.txn = self.env.begin(write=True)

    def close(self):
        self.txn.commit()
        self.env.close()
        self.txt_file.close()

# 本项目test时使用PNG图像输入，这个函数是用于制作所需的所有test PNG图像到一个目录下
def odv_test_png240_ForTest():
    f1 = open(os.path.join('/home/yl/bvqa360/data/', 'test_240ODV.txt'), "r")
    lines = f1.readlines()
    for num_ref in range(0, 108):
        line = lines[num_ref]
        # print(line)
        line_list = line.split()
        vid_id = line_list[1]
        print(vid_id)
        for im in os.listdir(os.path.join(png240_dir, vid_id)):
            print(im)
            tar = os.path.join(new_data_png240, vid_id) + '_' + im
            print(tar)
            src = os.path.join(png240_dir, vid_id, im)
            shutil.copyfile(src, tar)




#########################################################################################################################################################
##### Prepare any kinds of imformation and configurarion for the IQA/VQA/BVQA compared algrorithm #####
#########################################################################################################################################################

################################################
##### Dataset path for compared algorithms #####
################################################

saved_log = r'/home/yl/PSNR_BVQA360/'

# ./360tools_metric
# < Usage >
#   -c, --config [STRING] (optional)
#     : config file name
#   -o, --original [STRING]
#     : original file name
#   -r, --reconstruct [STRING]
#     : output file name
#   -w, --original_width [INTEGER]
#     : width of original image
#   -h, --original_height [INTEGER]
#     : height of original image
#   -l, --reconstructed_width [INTEGER] (optional)
#     : width of reconstructed image
#   -m, --reconstructed_height [INTEGER] (optional)
#     : height of reconstruncted image
#   -n, --num_frame [INTEGER] (optional)
#     : number of frames to be processed
#   -q, --qmetric [INTEGER]
#     : quality metric
#         1: PSNR
#         2: sperical PSNR(S-PSNR)
#         3: weighted sperical PSNR(WS-PSNR)
#         4: craster parabolic projection PSNR(CPP-PSNR)
#   -s, --sphere_file [STRING] (optional)
#     : sphere file; used for calculation of S-PSNR
#   -f, --original_proj_format [INTEGER] (optional)
#     : original projection format
#          1: ERP
#          2: ISP
#          3: CMP
#          4: OHP
#          5: TSP
#          6: CPP
#          7: SSP
#   -t, --reconstructed_proj_format [INTEGER] (optional)
#     : reconstructed projection format
#          1: ERP
#          2: ISP
#          3: CMP
#          4: OHP
#          5: TSP
#          6: CPP
#          7: SSP
#   -x, --color_space_orig [INTEGER] (optional)
#     : color space
#          1: YUV420 8-bit
#          2: YUV420 10-bit
#
#   -y, --color_space_ref [INTEGER] (optional)
#     : color space
#          1: YUV420 8-bit
#          2: YUV420 10-bit
#
#   -v, --verbose [FLAG] (optional)
#     : verbose output


### 1. PSNR/S-PSNE/WS-PSNR/CPP-PSNR
    # 执行：放在360tools/bin里面
def countPSNR():
    # spsnrtxt = '/home/liutie/Desktop/360tools-master/bin/sphere655362.txt'
    log_pth = os.path.join(saved_log,  'psnr_spsnr++_dmos.txt')

    f = open(os.path.join(dataset_utils, 'train.txt'), "r")
    lines = f.readlines()  # 读取全部内容 ，并以列表方式返回
    namelist = []
    i = 0
    for num_ref in range(0, 540):
        line = lines[num_ref]
        line_list = line.split()  # string to list
        vid_id = line_list[0]
        vid_id = int(vid_id)

        if vid_id in odv_test_id:
            # print('!!!!!!!!!!')
            # num+=1
            # print('the train number is {}'.format(num))
            imp_name = os.path.split(line_list[3])[-1]
            ref_name = os.path.split(line_list[2])[-1]

            group_name = 'Group' + imp_name[1]

            if num_ref >= (540 - 54):
                group_name = 'Group' + str(10)

            ref_pth = os.path.join(orig_data, group_name, 'Reference', ref_name)
            test_pth = os.path.join(orig_data, group_name, imp_name)

            w = int(line_list[5])
            h = int(line_list[6])
            num_frame = int(line_list[7])


            i += 1
            print('this is {}-th test odv ##############'.format(i))

            for metric in range(1, 5):
                if metric == 1:
                    cmd1 = r'./360tools_metric -w {} -h {} -n {} -f 1 -o {} -q {} -l {} -m {} -t 1 -r {} >> {}'  # -n: frames
                    cmd1 = cmd1.format(w, h, num_frame, ref_pth, str(metric), w, h, test_pth, log_pth)
                    os.system(cmd1)
                    print(cmd1)
                elif metric == 2:
                    cmd2 = r'./360tools_metric -w {} -h {} -n {} -f 1 -o {} -q {} -l {} -m {} -s sphere_655362.txt -t 1 -r {} >> {}'
                    cmd2 = cmd2.format(w, h, num_frame, ref_pth, str(metric), w, h, test_pth, log_pth)
                    os.system(cmd2)
                    print(cmd2)
                elif metric == 3:
                    cmd3 = r'./360tools_metric -w {} -h {} -n {} -f 1 -o {} -q {} -l {} -m {} -t 1 -r {} >> {}'  # -n: frames
                    cmd3 = cmd3.format(w, h, num_frame, ref_pth, str(metric), w, h, test_pth, log_pth)
                    os.system(cmd3)
                    print(cmd3)
                elif metric == 4:
                    cmd4 = r'./360tools_metric -w {} -h {} -n {} -f 1 -o {} -q {} -l {} -m {} -t 1 -r {} >> {}'
                    cmd4 = cmd4.format(w, h, num_frame, ref_pth, str(metric), w, h, test_pth, log_pth)
                    os.system(cmd4)
                    print(cmd4)



# extract the Y channel psnr..  shape:(960, 4)
def cal_Y_PSNR():
    psnr = np.loadtxt(os.path.join(saved_log,  'psnr_spsnr++_dmos.txt'), dtype=np.str)[:, 1].astype(np.float32).reshape((108, 4))
    #print(psnr.dtype)
    np.savetxt(os.path.join(saved_log,  'PSNR_S-PSNE_WS-PSNR_CPP-PSNR_dmos.txt') , psnr, fmt='%.04f')


### 1. PSNR/S-PSNE/WS-PSNR/CPP-PSNR
    # 执行：放在360tools/bin里面
def countPSNR_BIT():
    # spsnrtxt = '/home/liutie/Desktop/360tools-master/bin/sphere655362.txt'
    log_pth = os.path.join(saved_log,  'psnr_spsnr++_BIT_dmos4.txt')

    f = open(r'/home/yl/bvqa360/bit_dataset/score.txt', "r")
    lines = f.readlines()  # 读取全部内容 ，并以列表方式返回
    vid_dmos = []
    i = 0

    for idx in range(len(lines)):
        line = lines[idx]
        imp_name = line.split()[0]
        id = imp_name.split('-')[0]
        ref_vid = id + '-' + '100MB' + '.yuv'
        imp_vid = imp_name + '.yuv'
        ref_pth = os.path.join(data_BIT_ref_yuv240, ref_vid)

        test_pth = os.path.join(data_BIT_imp_yuv240, imp_vid)

        w = 480
        h = 240
        num_frame = 10

        i += 1
        print('this is {}-th test odv ##############'.format(i))

        for metric in range(1, 5):
            if metric == 1:
                cmd1 = r'./360tools_metric -w {} -h {} -n {} -f 1 -o {} -q {} -l {} -m {} -t 1 -r {} >> {}'  # -n: frames
                cmd1 = cmd1.format(w, h, num_frame, ref_pth, str(metric), w, h, test_pth, log_pth)
                os.system(cmd1)
                print(cmd1)
            elif metric == 2:
                cmd2 = r'./360tools_metric -w {} -h {} -n {} -f 1 -o {} -q {} -l {} -m {} -s sphere_655362.txt -t 1 -r {} >> {}'
                cmd2 = cmd2.format(w, h, num_frame, ref_pth, str(metric), w, h, test_pth, log_pth)
                os.system(cmd2)
                print(cmd2)
            elif metric == 3:
                cmd3 = r'./360tools_metric -w {} -h {} -n {} -f 1 -o {} -q {} -l {} -m {} -t 1 -r {} >> {}'  # -n: frames
                cmd3 = cmd3.format(w, h, num_frame, ref_pth, str(metric), w, h, test_pth, log_pth)
                os.system(cmd3)
                print(cmd3)
            elif metric == 4:
                cmd4 = r'./360tools_metric -w {} -h {} -n {} -f 1 -o {} -q {} -l {} -m {} -t 1 -r {} >> {}'
                cmd4 = cmd4.format(w, h, num_frame, ref_pth, str(metric), w, h, test_pth, log_pth)
                os.system(cmd4)
                print(cmd4)





# extract the Y channel psnr..  shape:(960, 4)
def cal_Y_PSNR_BIT():
    psnr = np.loadtxt(os.path.join(saved_log,  'psnr_spsnr++_BIT_dmos4.txt'), dtype=np.str)[:, 1].astype(np.float32).reshape((144, 4))
    #print(psnr.dtype)
    np.savetxt(os.path.join(saved_log,  'BIT_PSNR_S-PSNE_WS-PSNR_CPP-PSNR_dmos4.txt') , psnr, fmt='%.04f')


# odv_test_png240 = r'/media/yl/yl_8t/data_bvqa_lc_resize/data_vqa_yl240_png_test' # test 测试集 108个目录的png
# odv_ref_png240 = r'/media/yl/yl_8t/data_bvqa_lc_resize/data_vqa_yl240_png_ref'

def cal_WSSSIM():
    log_pth = os.path.join(saved_log, 'WS-SSIM_dmos.txt')

    f = open(os.path.join(dataset_utils, 'train.txt'), "r")
    lines = f.readlines()  # 读取全部内容 ，并以列表方式返回
    vid_dmos = []
    i = 0
    for num_ref in range(0, 540):
        line = lines[num_ref]
        line_list = line.split()  # string to list
        vid_id = line_list[0]
        vid_id = int(vid_id)

        if vid_id in odv_test_id:
            # print('!!!!!!!!!!')
            # num+=1
            # print('the train number is {}'.format(num))
            imp_name = os.path.split(line_list[3])[-1]
            ref_name = os.path.split(line_list[2])[-1]
            print(imp_name)
            num_frame = int(line_list[7])
            group_name = 'Group' + imp_name[1]

            if num_ref >= (540 - 54):
                group_name = 'Group' + str(10)

            imp_name = os.path.splitext(imp_name)[0]
            ref_name = os.path.splitext(ref_name)[0]

            ref_pth = os.path.join(odv_ref_png240, group_name, 'Reference', ref_name)
            print(ref_pth)
            test_pth = os.path.join(odv_test_png240, imp_name)

            frame_dmos = 0
            for test_frame in os.listdir(test_pth):
                test_frame_pth = os.path.join(test_pth, test_frame)
                ref_frame_pth = os.path.join(ref_pth, test_frame)

                test = cv2.imread(test_frame_pth, cv2.IMREAD_COLOR)
                test = cv2.cvtColor(test, cv2.COLOR_BGR2RGB).astype(np.uint8)
                #print(test.shape)
                ref = cv2.imread(ref_frame_pth, cv2.IMREAD_COLOR)
                #print(ref)
                ref = cv2.cvtColor(ref, cv2.COLOR_BGR2RGB).astype(np.uint8)

                test = cv2.cvtColor(test, cv2.COLOR_RGB2GRAY)
                ref= cv2.cvtColor(ref, cv2.COLOR_RGB2GRAY)

                # ref = np.array(Image.open(ref_frame_pth))
                # test = np.array(Image.open(test_frame_pth))


                frame_dmos = frame_dmos + ws_ssim(ref, test) / num_frame

            vid_dmos.append(frame_dmos)

    np.savetxt(log_pth, np.array(vid_dmos))


def cal_BIT_WSSSIM():

    log_pth = os.path.join(saved_log, 'WS-SSIM_BIT_dmos5.txt')

    f = open(r'/home/yl/bvqa360/bit_dataset/score.txt', "r")
    lines = f.readlines()  # 读取全部内容 ，并以列表方式返回
    vid_dmos = []
    i = 0
    interval = 300
    num_frame = 300 / interval
    for idx in range(len(lines)):
        line = lines[idx]
        imp_name = line.split()[0]
        id = imp_name.split('-')[0]
        ref_vid = id + '-' + '100MB'
        imp_vid = imp_name
        ref_pth = os.path.join(data_BIT_ref_png240, ref_vid)
        # print(ref_pth)
        test_pth = os.path.join(data_BIT_imp_png240, imp_vid)

        frame_dmos = 0
        for j in range(1, 301, interval):
            test_frame = '%03d'%j + '.png'
        #for test_frame in os.listdir(test_pth):
            test_frame_pth = os.path.join(test_pth, test_frame)
            ref_frame_pth = os.path.join(ref_pth, test_frame)

            test = cv2.imread(test_frame_pth, cv2.IMREAD_COLOR)
            test = cv2.cvtColor(test, cv2.COLOR_BGR2RGB).astype(np.uint8)
            # print(test.shape)
            ref = cv2.imread(ref_frame_pth, cv2.IMREAD_COLOR)
            # print(ref)
            ref = cv2.cvtColor(ref, cv2.COLOR_BGR2RGB).astype(np.uint8)

            test = cv2.cvtColor(test, cv2.COLOR_RGB2GRAY)
            ref = cv2.cvtColor(ref, cv2.COLOR_RGB2GRAY)

            # ref = np.array(Image.open(ref_frame_pth))
            # test = np.array(Image.open(test_frame_pth))

            frame_dmos = frame_dmos + ws_ssim(ref, test) / num_frame

        vid_dmos.append(frame_dmos)


    np.savetxt(log_pth, np.array(vid_dmos))




#########################################################################################################################################################
##### Prepare the dataset of BIT360video dataset[1] for validate the generalization ability of the IQA/VQA/BVQA compared algrorithms #####
#########################################################################################################################################################
# [1] SUBJECTIVE AND OBJECTIVE QUALITY ASSESSMENT OF PANORAMIC VIDEOS IN VIRTUAL REALITY ENVIRONMENTS,
#     Bo Zhang, lunzhe Zhao, Shu Yang, Yang Zhang, ling Wang*, Zesong Fei
########################################################################################################

################################################
##### Dataset path of BIT360video #####
################################################

#

data_BIT_imp_org = r'/media/yl/yl_8t/data_vqa_BIT/Video' # 存放原始数据库中的 失真视频的MP4或者MKV格式 (4096, 2048) 144个
data_BIT_ref_mp4_org = r'/media/yl/yl_8t/data_vqa_BIT/Ref' # 存放原始数据库中的 reference视频的MP4格式 (4096, 2048)
data_BIT_ref_yuv_org = r'/media/yl/yl_8t/data_vqa_BIT/Ref_yuv'  # 存放原始数据库中的 reference视频的yuv格式 (4096, 2048)


data_BIT_imp_mp4 = r'/media/yl/yl_8t/data_vqa_BIT/data_BIT_imp_mp4' # 存放所有BIT的失真视频的MP4格式 (4096, 2048)
data_BIT_imp_yuv = r'/media/yl/yl_8t/data_vqa_BIT/data_BIT_imp_yuv' # 存放所有BIT的失真视频的yuv格式 (4096, 2048)

data_BIT_imp_mp4240 = r'/media/yl/yl_8t/data_vqa_BIT/data_BIT_imp_mp4240' # 存放 resized 所有BIT的失真视频的MP4格式 (480, 240)
data_BIT_imp_yuv240 = r'/media/yl/yl_8t/data_vqa_BIT/data_BIT_imp_yuv240' # 存放 resized 所有BIT的失真视频的yuv格式 (480, 240)
data_BIT_imp_png240 = r'/media/yl/yl_8t/data_vqa_BIT/data_BIT_imp_png240' # 存放 resized 所有BIT的失真视频的png格式 (480, 240)

data_BIT_ref_mp4240 = r'/media/yl/yl_8t/data_vqa_BIT/data_BIT_ref_mp4240' # 存放 resized 所有BIT的reference视频的MP4格式 (480, 240)
data_BIT_ref_yuv240 = r'/media/yl/yl_8t/data_vqa_BIT/data_BIT_ref_yuv240' # 存放 resized 所有BIT的reference视频的yuv格式 (480, 240)
data_BIT_ref_png240 = r'/media/yl/yl_8t/data_vqa_BIT/data_BIT_ref_png240' # 存放 resized 所有BIT的reference视频的yuv格式 (480, 240)

# For compared algorithms

data_compare_BIT_test_png_BVQA360 = r'/media/yl/yl_8t/data_vqa_BIT/data_compare_BIT_test_png_BVQA360' # 将所有视频所有帧全部分解为PNG然后放到同一个目录下





ref_BIT_name = ['11-100MB', '12-100MB', '13-100MB', '14-100MB', '15-100MB', '16-100MB'] # The list of 6 reference videos
test_id = ['11','12','13','14','15','16'] # The test video ID
imp_types = ['6MB-265','10MB-264','10MB-V9','1.8MB-265','3MB-264','3MB-V9','300KB-265','500KB-264','500KB-V9','1.8MB-V9','6MB-V9','300KB-V9',
            '1.8MB-264','6MB-264','300KB-264','3MB-265','3MB-N2','3MB-N6','3MB-BLUR','10MB-265','10MB-BLUR','10MB-N2','10MB-N6','500KB-265'] # 24 distortion on BIT

info_BIT = r'/media/yl/yl_8t/data_vqa_BIT/bit_dataset/score.txt' # 1-st column: 11-3MB-265 ; 2-nd column: 16.9524533421962


# 1. 所有文件转化为mp4, including the format of MKV to MP4

def all_mkv2mp4_BIT():
    for vid in os.listdir(data_BIT_imp_org):
        if os.path.splitext(vid)[-1] == '.mkv':
            vid_in_pth = os.path.join(data_BIT_imp_org, vid)
            vid_out_pth = os.path.join(data_BIT_imp_mp4, os.path.splitext(vid)[0]+'.mp4')
            cmd = r'ffmpeg -i {} -vcodec copy -acodec aac {}'.format(vid_in_pth, vid_out_pth)
            os.system(cmd)

        else:
            shutil.copyfile(os.path.join(data_BIT_imp_org, vid), os.path.join(data_BIT_imp_mp4, vid))

# 2. 所有imp mp4 转化为 yuv 到 data_BIT_imp_yuv

def imp_BIT_mp4Toyuv():
    for vid in os.listdir(data_BIT_imp_mp4):
        vid_mp4 = os.path.join(data_BIT_imp_mp4, vid)
        vid_yuv = os.path.join(data_BIT_imp_yuv, os.path.splitext(vid)[0] + '.yuv')
        cmd = r'ffmpeg -i {} {}'.format(vid_mp4, vid_yuv)

        os.system(cmd)

# 3. 所有 imp yuv 4096 转化为 480 240 到 data_BIT_imp_yuv240

def imp_BIT_yuv2resize():
    for vid in os.listdir(data_BIT_imp_yuv):
        res = '4096x2048'
        res_lr = '480x240'
        org_ref_video = os.path.join(data_BIT_imp_yuv, vid)
        lr_ref_video = os.path.join(data_BIT_imp_yuv240, vid)
        cmd = r'ffmpeg -f rawvideo -s {} -pix_fmt yuv420p -i {} -s {} -c:v rawvideo {}'.format(res, org_ref_video,
                                                                                                res_lr, lr_ref_video)
        os.system(cmd)


# 4. 所有 imp yuv 480 240 转化为 imp mp4 480 240 到 data_BIT_imp_mp4240

def imp_BIT240_yuv2mp4():
    for vid in os.listdir(data_BIT_imp_yuv240):
        fps = 30
        ori_yuv = os.path.join(data_BIT_imp_yuv240, vid)
        test_mp4 = os.path.join(data_BIT_imp_mp4240, os.path.splitext(vid)[0] + '.mp4')

        cmd = 'ffmpeg -f rawvideo -pix_fmt yuv420p -s 480x240 -r {} -i {} ' \
              '-vcodec libx265 -x265-params lossless=1 -r {}  {}'.format(fps, ori_yuv, fps, test_mp4)
        os.system(cmd)

# 5. 所有 ref yuv 4096 转化为 480 240  到 data_BIT_ref_yuv240

def ref_BIT_yuv2resize():
    for vid in os.listdir(data_BIT_ref_yuv_org):
        res = '4096x2048'
        res_lr = '480x240'
        org_ref_video = os.path.join(data_BIT_ref_yuv_org, vid)
        lr_ref_video = os.path.join(data_BIT_ref_yuv240, vid)
        cmd = r'ffmpeg -f rawvideo -s {} -pix_fmt yuv420p -i {} -s {} -c:v rawvideo {}'.format(res, org_ref_video,
                                                                                                res_lr, lr_ref_video)
        os.system(cmd)

# 6. 所有 ref yuv 240 转化为 ref mp4 480 240 到 data_BIT_ref_mp4240

def ref_BIT240_yuv2mp4():
    for vid in os.listdir(data_BIT_ref_yuv240):
        fps = 30
        ori_yuv = os.path.join(data_BIT_ref_yuv240, vid)
        test_mp4 = os.path.join(data_BIT_ref_mp4240, os.path.splitext(vid)[0] + '.mp4')

        cmd = 'ffmpeg -f rawvideo -pix_fmt yuv420p -s 480x240 -r {} -i {} ' \
              '-vcodec libx265 -x265-params lossless=1 -r {}  {}'.format(fps, ori_yuv, fps, test_mp4)
        os.system(cmd)

# 7. 所有 imp yuv 240 转化为 imp png 240 到 data_BIT_imp_png240

def BIT_imp_yuv2png_240():
    for vid in os.listdir(data_BIT_imp_yuv240):
        name, ext = os.path.splitext(vid)
        vid_pth = os.path.join(data_BIT_imp_yuv240, vid)
        if not os.path.exists(os.path.join(data_BIT_imp_png240, name)):
            os.mkdir(os.path.join(data_BIT_imp_png240, name))
        out_dir = os.path.join(data_BIT_imp_png240, name, '%3d.png')
        cmd = 'ffmpeg -f rawvideo -pixel_format yuv420p -s 480x240 -i {} {}'.format(vid_pth, out_dir)
        os.system(cmd)

# 8. 所有 ref yuv 240 转化为 imp png 240 到 data_BIT_ref_png240

def BIT_ref_yuv2png_240():
    for vid in os.listdir(data_BIT_ref_yuv240):
        name, ext = os.path.splitext(vid)
        vid_pth = os.path.join(data_BIT_ref_yuv240, vid)
        if not os.path.exists(os.path.join(data_BIT_ref_png240, name)):
            os.mkdir(os.path.join(data_BIT_ref_png240, name))
        out_dir = os.path.join(data_BIT_ref_png240, name, '%3d.png')
        cmd = 'ffmpeg -f rawvideo -pixel_format yuv420p -s 480x240 -i {} {}'.format(vid_pth, out_dir)
        os.system(cmd)

# 9. 将 BIT 数据库中的所有png 整理到一个目录 为了我的模型 测试 准备 到 data_compare_BIT_test_png_BVQA360

def BIT_png_test_bvqa360():
    for vid in os.listdir(data_BIT_imp_png240):
        for img in os.listdir(os.path.join(data_BIT_imp_png240, vid)):
            src = os.path.join(data_BIT_imp_png240, vid, img)
            tar = os.path.join(data_compare_BIT_test_png_BVQA360, vid + '_' + img)
            shutil.copyfile(src, tar)








#########################################################################################################################################################
############## Compared algorithms preparation ##########################
#########################################################################################################################################################

##########################################  Blind visual quality assessment on 2D or 360 videos #########################################################

#### 1. NR-OVQA 2. VIDEVAL 3. V-MEON 4. VSFA 5. TLVQM 6. NSTSS-3D-MSCN 7. NSTSS-ST_Gabor 8. NSTSS-Plus ############

# 1. NR-OVQA   《Blind quality assessment of omnidirectional videos using spatio-temporal convolutional neural networks》, Optic 2021

# Train and Test:

# ## code: /home/yl/NR-OVQA ; data: /media/yl/yl_8t/data_bvqa_lc_resize/data_NROVQA_cmp_png or /media/yl/yl_8t/data_vqa_BIT/compared_algo_data_process;
# ## results: /media/yl/yl_8t/data_bvqa_lc_resize/NR-OVQA_models or /media/yl/yl_8t/data_bvqa_lc_resize/NR-OVQA_models/BIT_dataset_results

# (1) 制作训练测试时所需的matlab .mat文件：name.mat, newdmos1.mat, p7.mat, p8.mat。

# (2) 在matlab中进行处理 运行 pre_process.m (yl dataset) 或者 pre_process_BIT.m (BIT dataset)

# (3) name.mat: 所有视频的名称，共 540*6=3240 (包含训练和测试集) 个 或者 144*6=864 个 (只包含测试集)；
#     newdmos1.mat: 所有CMP视频的分数，共 540*6*18=58320 或者 144*6*18=15552；p7.mat: 432*6=2592, p8.mat: 108*6=648 (BIT: 144*6=864)
#     存放目录：yl: /home/yl/NR-OVQA/BOVQA/lc_data_new 或者 /home/yl/NR-OVQA/BOVQA/bit_data_new

# (4) 将所有视频转化为CMP格式，六个面。运行 /home/yl/NR-OVQA/ERP-CMP-master/erpToCmp.py or ero_ToCmp_BIT.py; 540*6=3240 个video 目录，每个视频下300张png。

# (5) 主程序：修改 WPFolder.py or WPFolder_BIT.py; Train:SCNN.py; Test: SCNN_test.py or SCNN_test_BIT.py


# 2. Videval   《UGC-VQA: Benchmarking Blind Video Quality Assessment for User Generated Content》, TIP 2021

# ## code: /home/yl/VIDEVAL-master; data: /media/yl/yl_8t/data_vqa_BIT/data_BIT_imp_yuv240
# ## results: /home/yl/VIDEVAL-master/results

# (1) 先准备lc数据库240的xls文件 用于feature提取：TEST_VIDEOS_metadata.xls or TEST_BIT_VIDEOS_metadata.xls
#     格式：列：[videoname, mos, width, height, pixfmt, framerate, nb_frames, bitdepth, bitrate]  108 or 144 行
# #     code: /home/yl/bvqa360/creat_lmdb.py中的 videval_make_csv_BIT240() 函数.
#
# # (2) 提取视频特征：demo_compute_VIDEVAL_light_feats.m 或者 demo_compute_BIT_VIDEVAL_light_feats.m，得到./features/TEST_VIDEOS_VIDEVAL_light720_6fps_feats.mat.
#
# # (3) 然后运行 demo_pred_MOS_pretrained_VIDEVAL_light.py 预测mos.
















if __name__ == '__main__':
    #countPSNR()
    #cal_Y_PSNR()

    #yuv2png_ref_240()

    #cal_WSSSIM()

    #all_mkv2mp4_BIT()

    #imp_BIT_mp4Toyuv()

    # imp_BIT_yuv2resize()
    #imp_BIT240_yuv2mp4()

    # ref_BIT_yuv2resize()
    # ref_BIT240_yuv2mp4()

    # BIT_imp_yuv2png_240()
    # BIT_ref_yuv2png_240()
    #BIT_png_test_bvqa360()

    cal_BIT_WSSSIM()
    #countPSNR_BIT()

    #cal_Y_PSNR_BIT()