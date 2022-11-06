'''
Updated: 11-04-22
Author @ shahzaib (AI Dev)
'''

import os

classes = ['nfpa']
no_of_classes = len(classes)
window_size = [416,416]

# Please change dataset folder name
# dataset_folder_name = 'sws_n2'
dataset_folder_name = os.getcwd().split('/')[-1]

DATASET_DIR_train = './train/'
DIR_PATH_train = dataset_folder_name+'/train/'
FILE_PATH_train = './train.txt'

img_files_train = os.listdir(DATASET_DIR_train)

f = open(FILE_PATH_train, 'w')
for img_file in img_files_train:
    if img_file.endswith('.jpg'):
        f.write(f'{DIR_PATH_train}{img_file}\n')
f.close()


DATASET_DIR_valid = './valid/'
DIR_PATH_valid = dataset_folder_name+'/valid/'
FILE_PATH_valid = './valid.txt'

img_files_valid = os.listdir(DATASET_DIR_valid)

f = open(FILE_PATH_valid, 'w')
for img_file in img_files_valid:
    if img_file.endswith('.jpg'):
        f.write(f'{DIR_PATH_valid}{img_file}\n')
f.close()


FILE_PATH_objdata = './obj.data'
f = open(FILE_PATH_objdata, 'w')
f.write('classes = '+str(no_of_classes)+'\n')
f.write('train = '+dataset_folder_name+'/'+'train.txt\n')
f.write('valid = '+dataset_folder_name+'/'+'valid.txt\n')
f.write('names = '+dataset_folder_name+'/'+'obj.names\n')
f.write('backup = '+dataset_folder_name+'/'+'backup/\n')
f.close()


FILE_PATH_objnames = './obj.names'
f = open(FILE_PATH_objnames, 'w')
for classname in classes:
    f.write(classname+'\n')
f.close()


FILE_PATH_ach = '../anchors.sh'
f = open(FILE_PATH_ach, 'w')
f.write('./darknet detector calc_anchors '+dataset_folder_name+'/obj.data -num_of_clusters 9 -width '+str(window_size[0])+' -height '+str(window_size[1])+'\n')
f.close()


FILE_PATH_trainsh = '../train.sh'
f = open(FILE_PATH_trainsh, 'w')
f.write('./darknet detector train '+dataset_folder_name+'/obj.data '+dataset_folder_name+'/yolov4.cfg yolov4.conv.137 -map\n')
f.close()




