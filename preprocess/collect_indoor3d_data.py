import os
import sys
from indoor3d_util import collect_point_label

anno_paths = [line.rstrip() for line in open('/home/g111056119/Documents/7111056426/Reptile-Pytorch/datasets/S3DIS/meta/anno_paths.txt')]
DATA_PATH='/home/g111056119/Documents/7111056119/Pointnet_Pointnet2_pytorch/data/s3dis/Stanford3dDataset_v1.2_Aligned_Version'
anno_paths = [os.path.join(DATA_PATH, p) for p in anno_paths]

# 只標注和 scannet 相同的類別 >> 和在一起做 meta learning 適應到不同資料集 
output_folder = "/media/g111056119/SP PHD U3/datasets/S3DIS/npy_9classes/"
if not os.path.exists(output_folder):
    os.mkdir(output_folder)

# Note: there is an extra character in the v1.2 data in Area_5/hallway_6. It's fixed manually.
for anno_path in anno_paths:
    print(anno_path)
    """/home/g111056119/Documents/7111056119/Pointnet_Pointnet2_pytorch/data/s3dis/Stanford3dDataset_v1.2_Aligned_Version/Area_1/conferenceRoom_1/Annotations"""
    try:
        elements = anno_path.split('/')
        """
        ['', 'home', 'g111056119', 'Documents', '7111056119', 'Pointnet_Pointnet2_pytorch', 'data', 's3dis', 'Stanford3dDataset_v1.2_Aligned_Version', 'Area_1', 'conferenceRoom_1', 'Annotations']
        """ 
        out_filename = elements[-2]+'_'+elements[-3]+'.npy' # Area_1_hallway_1.npy >> hallway_1_Area_1.npy
        collect_point_label(anno_path, os.path.join(output_folder, out_filename), 'numpy')
    except:
        print(anno_path, 'ERROR!!')
