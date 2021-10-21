import os
ref_yuv_dir = r'/media/yl/yl_8t/data_vqa_BIT/'

all_files = os.listdir('./Ref/')
if not os.path.exists(os.path.join(ref_yuv_dir, 'Ref_yuv')):
    os.mkdir(os.path.join(ref_yuv_dir, 'Ref_yuv'))
for file in all_files:
    os.system('ffmpeg -i ./Ref/' +file +' ./Ref_yuv/'+ file[:-4] + '.yuv')
