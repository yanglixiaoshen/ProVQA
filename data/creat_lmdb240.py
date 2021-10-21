
import argparse
from os import path as osp
import cv2
#from basicsr.utils import scandir
#from basicsr.utils.lmdb_util import make_lmdb_from_imgs
import cv2
import lmdb
import sys
from multiprocessing import Pool
from os import path as osp
from tqdm import tqdm
import os
import random
import numpy as np
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

odvDataset_pth = r'/media/yl/yl_8t/data_vqa_yl720_png/'
odvLmdb_pth = r'/media/yl/yl_8t/data_vqa_yl720_lmdb/odv720-vqa_dataset.lmdb'
odvLmdb_train_pth = r'/media/yl/yl_8t/data_vqa_yl720_lmdb_train/odv720-vqa_dataset_train.lmdb'
meta_info_pth = r'/media/yl/yl_8t/data_vqa_yl720_lmdb/odv720-vqa_dataset.lmdb/meta_info.txt'
odv_train_id = [0, 1 ,2, 5 ,6 ,7, 8 ,9, 10, 11, 13, 14, 15, 16, 18, 20, 22, 24, 25, 26, 27, 29, 31 ,32, 33, 34, 35, 36, 37, 38, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 59]
odv_test_id = [3, 4 ,12, 17, 19 ,21 ,23 ,28, 30, 39, 40, 58]
data_pth = r'/home/yl/bvqa360/data'



odvDataset_pth1 = r'/media/yl/yl_8t/data_vqa_yl480_png/'
odvLmdb_train_pth1 = r'/media/yl/yl_8t/data_vqa_yl480_lmdb_train/odv480-vqa_dataset_train.lmdb'
odv_train_png = r'/media/yl/yl_8t/data_vqa_yl480_png_train1'
odv_test_png = r'/media/yl/yl_8t/data_vqa_yl480_png_test1'

dataset_720= r'/media/yl/yl_8t/data_vqa_yl480'
png_dir = r'/media/yl/yl_8t/data_vqa_yl480_png'

dataset_240= r'/media/yl/yl_8t/data_bvqa_lc_resize/data_vqa_yl240'
png240_dir = r'/media/yl/yl_8t/data_bvqa_lc_resize/data_vqa_yl240_png'
odv_train240_png = r'/media/yl/yl_8t/data_bvqa_lc_resize/data_vqa_yl240_png_train'
odvLmdb_train_pth2 = r'/media/yl/yl_8t/data_bvqa_lc_resize/data_vqa_yl240_lmdb_train/odv240-vqa_dataset_train.lmdb'

odv_test_png240 = r'/media/yl/yl_8t/data_bvqa_lc_resize/data_vqa_yl240_png_test1'



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


import shutil
def shutil_train_test_png():
    f = open(os.path.join(data_pth, 'train.txt'), "r")
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


def create_lmdb_for_odv_train480_new():
    img_path_list, keys = prepare_keys_reds(odv_train240_png)

    make_lmdb_from_imgs(odv_train240_png, odvLmdb_train_pth2, img_path_list, keys, multiprocessing_read=True)

def create_lmdb_for_odv_train480():

    f = open(os.path.join(data_pth, 'train.txt'), "r")
    lines = f.readlines()  # 读取全部内容 ，并以列表方式返回

    vid_all_list = []
    keys_all_list = []
    #num = 1
    for num_ref in range(0, 540):
        line = lines[num_ref]
        #print(line)
        line_list = line.split()
        vid_id = line_list[0]
        vid_id = int(vid_id)
        #print(vid_id)
        if vid_id in odv_train_id:
            #print('!!!!!!!!!!')
            #num+=1
            #print('the train number is {}'.format(num))
            vid_name = os.path.split(line_list[3])[-1].split('.yuv')[0]

            #print(vid_name)
            folder_path = os.path.join(odvDataset_pth1, vid_name)
            img_path_list, keys = prepare_keys_reds(folder_path)
            vid_all_list += img_path_list
            keys_all_list += keys
    make_lmdb_from_imgs('/', odvLmdb_train_pth1, vid_all_list, keys_all_list, multiprocessing_read=True)

# ODV-VQA train test 的信息txt, [ID, VID_NAME, DMOS, FRAME_NUM, H, W]
def create_odv_train_test_txt1():
    f = open(os.path.join(data_pth, 'train.txt'), "r")
    lines = f.readlines()  # 读取全部内容 ，并以列表方式返回

    h = 240
    w = 480
    # num = 1

    f1 = open(os.path.join(data_pth, 'train_240ODV.txt'), 'w')
    f2 = open(os.path.join(data_pth, 'test_240ODV.txt'), 'w')
    for num_ref in range(0, 540):
        line = lines[num_ref]
        # print(line)
        line_list = line.split()
        vid_id = line_list[0]
        vid_id = int(vid_id)
        if vid_id in odv_train_id:
            l1 = vid_id # id
            #l2 = line_list[3].split(os.sep)[-1].split('.')[0] # name
            l2 = os.path.split(line_list[3])[-1].split('.yuv')[0]
            l3 = line_list[4] # dmos
            l4 = line_list[7] # frame_num
            l5 = h
            l6 = w
            line1 = '{} {} {} {} {} {}\n'.format(l1, l2, l3 , l4, l5, l6)
            f1.writelines(line1)
        else:
            l1 = vid_id  # id
            l2 = os.path.split(line_list[3])[-1].split('.yuv')[0]  # name

            l3 = line_list[4]  # dmos
            l4 = line_list[7]  # frame_num
            l5 = h
            l6 = w
            line2 = '{} {} {} {} {} {}\n'.format(l1, l2, l3, l4, l5, l6)
            f2.writelines(line2)
    f1.close()
    f2.close()




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

# def yuv2png():
#     for group in os.listdir(dataset_720):
#         group_pth = os.path.join(dataset_720, group)
#         for vid in os.listdir(group_pth):
#             name, ext = os.path.splitext(vid)
#             vid_pth = os.path.join(dataset_720, group, vid)
#             if not os.path.exists(os.path.join(png_dir, name)):
#                 os.mkdir(os.path.join(png_dir, name))
#             out_dir = os.path.join(png_dir, name, '%3d.png')
#             cmd = 'ffmpeg -f rawvideo -pixel_format yuv420p -s 720x360 -i {} {}'.format(vid_pth, out_dir)
#             os.system(cmd)


def creat_medical_image_info():
    dir = r'/media/yl/yl_8t/medical_test/test'
    f = open(os.path.join(r'/media/yl/yl_8t/medical_test', 'info.txt'), 'w')

    for patient_dir in os.listdir(dir):
        p_pth = os.path.join(dir, patient_dir)

        img_pth = os.path.join(p_pth, 'blur')
        gt_pth = os.path.join(p_pth, 'GT')
        quality_pth = os.path.join(p_pth, 'quality.txt')
        print(img_pth, quality_pth)

        score = np.loadtxt(quality_pth, dtype=str)
        for imp in os.listdir(img_pth):
            imp_pth = os.path.join(img_pth, imp)
            line = '{} {}\n'.format(imp_pth, score)
            f.writelines(line)
    f.close()


def creat_medical_image_info1():
    dir = r'/media/yl/yl_8t/medical_test/test'
    f = open(os.path.join(r'/media/yl/yl_8t/medical_test', 'd_info.txt'), 'w')

    for patient_dir in os.listdir(dir):
        p_pth = os.path.join(dir, patient_dir)
        p_s_pth = r'D:\\test' + '\\' + patient_dir

        img_pth = os.path.join(p_pth, 'blur')
        img_s_pth = p_s_pth + '\\' + 'blur'
        gt_pth = os.path.join(p_pth, 'GT')
        quality_pth = os.path.join(p_pth, 'quality.txt')
        print(img_pth, quality_pth)

        score = np.loadtxt(quality_pth, dtype=str)
        for imp in os.listdir(img_pth):
            imp_pth1 = img_s_pth + '\\' +  imp
            line = '{} {}\n'.format(imp_pth1, int(score))
            f.writelines(line)
    f.close()








if __name__ == '__main__':
    #yuv2png_240()
    #shutil_train_test_png()
    #create_lmdb_for_odv_train480_new()
    #create_odv_train_test_txt1()
    #create_lmdb_for_odv_train480()

    #create_odv_test_txt()
    #create_odv_testALL_txt()
    # odvDataset_pth1 = r'/media/yl/yl_8t/data_vqa_yl480_png/'
    # prepare_keys_reds(odvDataset_pth1)

    #creat_medical_image_info()
    creat_medical_image_info1()


