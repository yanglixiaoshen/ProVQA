import os
import scipy.io as sio

ref_list = ['11-100MB','12-100MB','13-100MB','14-100MB','15-100MB','16-100MB']
all_list = os.listdir('./Video/')
score = sio.loadmat('data.mat')['data'][10:,:]
print (score.shape)
lin_name = ['11','12','13','14','15','16']
col_name = ['6MB-265','10MB-264','10MB-V9','1.8MB-265','3MB-264','3MB-V9','300KB-265','500KB-264','500KB-V9','1.8MB-V9','6MB-V9','300KB-V9',
            '1.8MB-264','6MB-264','300KB-264','3MB-265','3MB-N2','3MB-N6','3MB-BLUR','10MB-265','10MB-BLUR','10MB-N2','10MB-N6','500KB-265']
frames = [21,66,111,156,201,246,291]

f = open('score.txt','wt')

for ref in ref_list:
    index = ref[:2]
    for item in all_list:
        print (item)
        if (item[:2]==index):
            for i in range(len(lin_name)):
                if (index==lin_name[i]):
                    lin = i
            if (len(item.split('-'))==3):
                name = item.split('-')[1]+'-'+item.split('-')[2].split('.')[0]
            else:
                name = item.split('-')[1][:-4]+'-'+'264'
            for j in range(len(col_name)):
                if (name==col_name[j]):
                    col = j
            sc = score[lin,col]
            print (name)
            print ('----')
            '''
            for frame in frames:
                output = index+' 0'+' /media/tatsu/Extra/BITVideo/Ref_yuv/' + ref + '.yuv'+' /media/tatsu/Extra/BITVideo/Video_yuv/' + item[:-4] + '.yuv ' + str(sc/100)+ ' 4096 2048 ' + str(frame)+'\n'
                f.write(output)
            '''
            output =  item[:-4] + ' ' + str(sc) + '\n'
            f.write(output)
